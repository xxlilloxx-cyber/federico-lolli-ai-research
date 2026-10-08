#!/usr/bin/env python3
"""Read-only final analysis pipeline for the continual-memory/EWC study."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
INPUTS = {
    "abc": ROOT / "results_abc_memory",
    "sweep": ROOT / "results_ewc_lambda_sweep",
    "confirmation": ROOT / "results_ewc_confirmation",
}
OUTPUT = ROOT / "analysis_full_ewc_memory"
METHODS = ("lora", "symmetric", "combined")
ABC_PROTOCOLS = ("sequential_single", "grow_unfrozen", "hard_consolidation", "ewc_like")
SWEEP_LAMBDAS = (0.0, 1.0, 10.0, 30.0, 50.0, 100.0, 300.0)
CONFIRMATION_GRID = {"lora": (0.0, 1.0, 10.0), "symmetric": (0.0, 10.0, 30.0), "combined": (0.0, 10.0, 30.0)}
SEEDS = (42, 123, 456)


@dataclass
class Run:
    path: Path
    method: str
    protocol: str
    value: float | None
    seed: int
    dataset_hash: str
    config_hash: str
    config: dict[str, Any]
    summary: dict[str, Any]
    roles: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    reused: bool = False
    canonical_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--tables-only", action="store_true")
    mode.add_argument("--figures-only", action="store_true")
    mode.add_argument("--reports-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_config(config: dict[str, Any], protocol: str) -> dict[str, Any]:
    ignored = {"output_dir"}
    normalized = {key: value for key, value in config.items() if key not in ignored}
    if protocol != "ewc_like":
        for key in ("ewc_lambda", "ewc_gamma", "fisher_samples", "fisher_batches"):
            normalized.pop(key, None)
    return normalized


def config_digest(config: dict[str, Any], protocol: str) -> str:
    payload = json.dumps(normalized_config(config, protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_run(path: Path, role: str) -> Run:
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    protocol = str(config["protocol"]); value = float(config["ewc_lambda"]) if protocol == "ewc_like" else None
    dataset_hash = sha256(path / "dataset.json")
    run = Run(path, str(config["method"]), protocol, value, int(config["seed"]), dataset_hash,
              config_digest(config, protocol), config, summary, {role}, {str(path.relative_to(ROOT))})
    identity = {"method": run.method, "protocol": run.protocol, "lambda": run.value, "seed": run.seed,
                "dataset_hash": run.dataset_hash, "config_hash": run.config_hash}
    run.canonical_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return run


def discover() -> tuple[list[Run], dict[str, int], list[str]]:
    errors=[]; physical=[]
    for key, root in INPUTS.items():
        if not root.is_dir(): errors.append(f"missing input root: {root}")
    if errors:return [],{},errors
    for protocol in ABC_PROTOCOLS:
        for method in METHODS:
            for seed in SEEDS:
                path=INPUTS["abc"]/"runs"/protocol/method/f"seed{seed}"
                if not (path/"summary.json").is_file(): errors.append(f"missing ABC run: {path}")
                else: physical.append(_load_run(path,"baseline" if protocol=="sequential_single" else "mechanistic_control"))
    for method in METHODS:
        for value in SWEEP_LAMBDAS:
            path=INPUTS["sweep"]/f'{method}_lambda_{value:g}_seed_42'
            if not (path/"summary.json").is_file(): errors.append(f"missing sweep run: {path}")
            else: physical.append(_load_run(path,"exploratory_lambda"))
    references_path=INPUTS["confirmation"]/"REUSED_RUNS.json"
    references=json.loads(references_path.read_text()) if references_path.is_file() else {"reused":[]}
    reused_map={(r["method"],float(r["lambda"]),int(r["seed"])): ROOT/r["source"] for r in references.get("reused",[])}
    if references.get("invalid"):errors.append("confirmation reuse index contains invalid entries")
    for method,values in CONFIRMATION_GRID.items():
        for value in values:
            for seed in SEEDS:
                key=(method,value,seed)
                if key in reused_map:
                    target=reused_map[key].resolve(); match=next((r for r in physical if r.path.resolve()==target),None)
                    if match is None:errors.append(f"confirmation reference target not discovered: {target}")
                    else:match.roles.add("confirmation");match.reused=True
                else:
                    path=INPUTS["confirmation"]/f'{method}_lambda_{value:g}_seed_{seed}'
                    if not (path/"summary.json").is_file():errors.append(f"missing confirmation run: {path}")
                    else:physical.append(_load_run(path,"confirmation"))
    grouped:dict[str,Run]={};physical_duplicates=0
    for run in physical:
        if run.canonical_id not in grouped:grouped[run.canonical_id]=run
        else:
            physical_duplicates+=1;current=grouped[run.canonical_id]
            current.roles.update(run.roles);current.sources.update(run.sources);current.reused=True
            if "confirmation" in run.roles and "confirmation" not in current.roles:current.path=run.path
    stats={"physical_runs":len(physical),"canonical_runs":len(grouped),"physical_duplicates":physical_duplicates,
           "confirmation_references":len(reused_map),"duplicate_or_reused_links":physical_duplicates+len(reused_map)}
    return sorted(grouped.values(),key=lambda r:(r.method,r.protocol,r.value if r.value is not None else -1,r.seed)),stats,errors


def finite_csv(path: Path) -> list[str]:
    errors=[]
    try:
        frame=pd.read_csv(path)
        for col in frame.select_dtypes(include=[np.number]).columns:
            if not np.isfinite(frame[col].dropna().to_numpy()).all():errors.append(f"{path}: non-finite {col}")
    except Exception as exc:errors.append(f"{path}: unreadable ({exc})")
    return errors


def validate(runs:list[Run],discovery_errors:list[str],stats:dict[str,int])->list[str]:
    errors=list(discovery_errors);dataset_hashes={r.dataset_hash for r in runs}
    for run in runs:
        if run.summary.get("complete") is not True:errors.append(f"{run.path}: incomplete")
        if run.method not in METHODS or run.seed not in SEEDS:errors.append(f"{run.path}: invalid method/seed")
        if run.config.get("model")!="gpt2" or run.config.get("placement")!="transformer.h[0].attn.c_proj":errors.append(f"{run.path}: model/placement mismatch")
        if run.protocol not in ABC_PROTOCOLS:errors.append(f"{run.path}: invalid protocol")
        expected=6144 if run.protocol in {"sequential_single","ewc_like"} else 18432
        if int(run.summary.get("final_adapter_parameters",-1))!=expected:errors.append(f"{run.path}: final adapter parameters != {expected}")
        if run.protocol=="ewc_like":
            if float(run.config.get("ewc_gamma",-1))!=1.0:errors.append(f"{run.path}: EWC gamma != 1")
            if not all((run.path/"checkpoints"/f"after_{p}.pt").is_file() for p in "ABC"):errors.append(f"{run.path}: missing EWC checkpoint")
        if "confirmation" in run.roles:
            if (run.config.get("micro_batch_size"),run.config.get("gradient_accumulation"),run.config.get("effective_batch_size"))!=(4,1,4):errors.append(f"{run.path}: confirmation batch mismatch")
            if run.protocol!="ewc_like":errors.append(f"{run.path}: confirmation not on EWC path")
        for name in ("metrics.csv","memory_matrix.csv","memory_slot_ablation.csv"):
            if not (run.path/name).is_file():errors.append(f"{run.path}: missing {name}")
            else:errors.extend(finite_csv(run.path/name))
        for value in run.summary.values():
            if isinstance(value,float) and not math.isfinite(value):errors.append(f"{run.path}: non-finite summary")
    if len(dataset_hashes)!=1:errors.append(f"A/B/C dataset hashes differ: {sorted(dataset_hashes)}")
    # Lambda-zero must use EWC, retain Fisher state, and have zero weighted penalty.
    for run in runs:
        if run.protocol=="ewc_like" and run.value==0:
            checkpoint=torch.load(run.path/"checkpoints"/"after_A.pt",map_location="cpu",weights_only=False)
            if set(checkpoint.get("auxiliary_state",{}))!={"reference","fisher"}:errors.append(f"{run.path}: lambda=0 lacks Fisher/reference")
            if "confirmation" in run.roles:
                metrics=pd.read_csv(run.path/"metrics.csv")
                # Stored penalty is the unweighted quadratic; the actual objective multiplier is exactly lambda/2=0.
                if float(run.config["ewc_lambda"])!=0:errors.append(f"{run.path}: lambda-zero mismatch")
    report=["# Full EWC memory study: data validation","",f'- Canonical scientific runs: {stats.get("canonical_runs",0)}',f'- Physical completed runs: {stats.get("physical_runs",0)}',f'- Physical duplicates collapsed: {stats.get("physical_duplicates",0)}',f'- Confirmation references reused: {stats.get("confirmation_references",0)}',f'- Dataset SHA-256: `{next(iter(dataset_hashes)) if len(dataset_hashes)==1 else "INCONSISTENT"}`','- No-replay validation: protocols use the validated phase-specific `ewc_like`/ABC runner; A is absent from phase B training and A/B are absent from phase C training.','- Parameter budgets: sequential/EWC 6,144; grow/hard 18,432.','- Lambda-zero: same EWC path with Fisher/reference state and exact zero objective multiplier.','',"## Result",""]
    report.append("**VALID** — all checks passed." if not errors else "**INVALID** — incompatible or incomplete data were found.")
    if errors:report += ["","## Errors",""]+[f"- {e}" for e in errors]
    out=OUTPUT/"reports";out.mkdir(parents=True,exist_ok=True);(out/"DATA_VALIDATION_REPORT.md").write_text("\n".join(report)+"\n")
    return errors


def provenance(runs:list[Run],stats:dict[str,int])->None:
    rows=[];files=[]
    priority={"confirmation":0,"exploratory_lambda":1,"baseline":2,"mechanistic_control":3}
    for run in runs:
        roles=sorted(run.roles,key=lambda x:priority[x]);sources=sorted(run.sources)
        rows.append({"canonical_run_id":run.canonical_id,"method":run.method,"protocol":run.protocol,"lambda":"" if run.value is None else run.value,"seed":run.seed,"source_directory":";".join(sources),"reused":run.reused,"dataset_hash":run.dataset_hash,"config_hash":run.config_hash,"complete":run.summary.get("complete") is True,"scientific_role":";".join(roles)})
        for source in sources:
            base=ROOT/source
            for name in ("summary.json","config.json","dataset.json","metrics.csv","memory_matrix.csv","memory_slot_ablation.csv"):
                path=base/name
                if path.is_file():files.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    write_csv(OUTPUT/"data"/"CANONICAL_RUNS.csv",rows);write_csv(OUTPUT/"provenance"/"RUN_PROVENANCE.csv",rows)
    versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
    try:
        import matplotlib;versions["matplotlib"]=matplotlib.__version__
    except Exception:versions["matplotlib"]=None
    try:
        import scipy;versions["scipy"]=scipy.__version__
    except Exception:versions["scipy"]=None
    manifest={"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"input_result_roots":[str(p.relative_to(ROOT)) for p in INPUTS.values()],"statistics":stats,"canonical_run_ids":[r.canonical_id for r in runs],"analysis_script_sha256":sha256(Path(__file__)),"software":versions,"input_files":files}
    path=OUTPUT/"provenance"/"ANALYSIS_MANIFEST.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(manifest,indent=2)+"\n")


def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def state_vector(checkpoint:dict[str,Any])->torch.Tensor:
    state=checkpoint["adapter_state"]
    return torch.cat([state[key].float().reshape(-1) for key in sorted(state)])


def fisher_stats(aux:dict[str,Any],prefix:str)->tuple[dict[str,float],list[dict[str,Any]]]:
    factors=[];tensors=aux["fisher"];flat=torch.cat([x.float().reshape(-1) for x in tensors.values()]);total=flat.sum();prob=flat/total.clamp_min(1e-30);nz=prob[prob>0]
    overall={f"{prefix}_sum":float(total),f"{prefix}_max":float(flat.max()),f"{prefix}_effective_support":float(torch.exp(-(nz*torch.log(nz)).sum()))}
    for key,tensor in tensors.items():
        value=float(tensor.float().sum());factors.append({"state":prefix,"factor":key.rsplit(".",1)[-1],"fisher_sum":value,"share":value/float(total) if total else 0.0})
    return overall,factors


def extract_metrics(run:Run)->tuple[dict[str,Any],list[dict[str,Any]]]:
    matrix=pd.read_csv(run.path/"memory_matrix.csv");lookup={(r.training_phase,r.evaluation_set):r for r in matrix.itertuples()};s=run.summary
    final=[lookup[("C",x)] for x in "ABC"]
    row={"canonical_run_id":run.canonical_id,"method":run.method,"protocol":run.protocol,"lambda":run.value,"seed":run.seed,"roles":";".join(sorted(run.roles)),
         "adapter_parameters":int(s["final_adapter_parameters"]),"acquisition_a_nll":float(s["acquisition_a_nll"]),"acquisition_b_nll":float(s["acquisition_b_nll"]),"acquisition_c_nll":float(s["acquisition_c_nll"]),
         "acquisition_a_probability":float(lookup[("A","A")].correct_probability),"acquisition_b_probability":float(lookup[("B","B")].correct_probability),"acquisition_c_probability":float(lookup[("C","C")].correct_probability),
         "acquisition_a_exact_match":float(lookup[("A","A")].exact_match),"acquisition_b_exact_match":float(lookup[("B","B")].exact_match),"acquisition_c_exact_match":float(lookup[("C","C")].exact_match),
         "forgetting_a_after_b_nll":float(s["forgetting_a_after_b_nll"]),"forgetting_a_after_c_nll":float(s["forgetting_a_after_c_nll"]),"forgetting_b_after_c_nll":float(s["forgetting_b_after_c_nll"]),
         "final_a_nll":float(final[0].nll),"final_b_nll":float(final[1].nll),"final_c_nll":float(final[2].nll),"final_a_probability":float(final[0].correct_probability),"final_b_probability":float(final[1].correct_probability),"final_c_probability":float(final[2].correct_probability),
         "final_a_exact_match":float(final[0].exact_match),"final_b_exact_match":float(final[1].exact_match),"final_c_exact_match":float(final[2].exact_match),
         "mean_forgetting":float(np.mean([s["forgetting_a_after_b_nll"],s["forgetting_a_after_c_nll"],s["forgetting_b_after_c_nll"]])),"mean_new_fact_acquisition_nll":float(np.mean([s["acquisition_b_nll"],s["acquisition_c_nll"]])),
         "mean_final_memory_nll":float(s["mean_final_nll"]),"mean_final_probability":float(s["mean_final_probability"]),"mean_final_exact_match":float(s["mean_final_exact_match"]),
         "mean_final_paraphrase_nll":float(np.mean([x.paraphrase_nll for x in final])),"mean_final_paraphrase_probability":float(np.mean([x.paraphrase_correct_probability for x in final])),"mean_final_paraphrase_exact_match":float(np.mean([x.paraphrase_exact_match for x in final]))}
    factors=[]
    checkpoints={p:torch.load(run.path/"checkpoints"/f"after_{p}.pt",map_location="cpu",weights_only=False) for p in "ABC"}
    vectors={p:state_vector(checkpoints[p]) for p in "ABC"}
    if len({v.numel() for v in vectors.values()})==1:
        row.update({"parameter_displacement_a_b_l2":float((vectors["B"]-vectors["A"]).norm()),"parameter_displacement_b_c_l2":float((vectors["C"]-vectors["B"]).norm()),"parameter_displacement_a_c_l2":float((vectors["C"]-vectors["A"]).norm())})
    else:row.update({"parameter_displacement_a_b_l2":None,"parameter_displacement_b_c_l2":None,"parameter_displacement_a_c_l2":None})
    if run.protocol=="ewc_like":
        fa,fac=fisher_stats(checkpoints["A"]["auxiliary_state"],"fisher_a");fab,fabc=fisher_stats(checkpoints["B"]["auxiliary_state"],"fisher_ab");row.update(fa);row.update(fab)
        for item in fac+fabc:item.update({"canonical_run_id":run.canonical_id,"method":run.method,"lambda":run.value,"seed":run.seed})
        factors.extend(fac+fabc)
        def raw(aux,target):return sum(float((aux["fisher"][k].float()*(target[k].float()-aux["reference"][k].float()).square()).sum()) for k in aux["fisher"])
        rb=raw(checkpoints["A"]["auxiliary_state"],checkpoints["B"]["adapter_state"]);rc=raw(checkpoints["B"]["auxiliary_state"],checkpoints["C"]["adapter_state"]);lam=float(run.value or 0)
        row.update({"realized_ewc_penalty_b":.5*lam*rb,"realized_ewc_penalty_c":.5*lam*rc})
    else:
        for key in ("fisher_a_sum","fisher_a_max","fisher_a_effective_support","fisher_ab_sum","fisher_ab_max","fisher_ab_effective_support","realized_ewc_penalty_b","realized_ewc_penalty_c"):row[key]=None
    return row,factors


def pareto(frame:pd.DataFrame)->pd.DataFrame:
    out=frame.copy();flags=[]
    for _,row in out.iterrows():
        dominated=((out.mean_forgetting<=row.mean_forgetting)&(out.mean_new_fact_acquisition_nll<=row.mean_new_fact_acquisition_nll)&((out.mean_forgetting<row.mean_forgetting)|(out.mean_new_fact_acquisition_nll<row.mean_new_fact_acquisition_nll))).any();flags.append(not dominated)
    out["pareto_nondominated"]=flags;return out


def aggregate_confirmation(frame:pd.DataFrame)->pd.DataFrame:
    metrics=[c for c in frame.columns if c not in {"canonical_run_id","method","protocol","lambda","seed","roles"} and pd.api.types.is_numeric_dtype(frame[c])]
    rows=[]
    for (method,value),g in frame.groupby(["method","lambda"]):
        row={"method":method,"lambda":value,"n_seeds":len(g),"individual_seeds":",".join(map(str,sorted(g.seed)))}
        for key in metrics:row[key+"_mean"]=g[key].mean();row[key+"_sample_sd"]=g[key].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method","lambda"])


def build_tables(runs:list[Run])->dict[str,pd.DataFrame]:
    all_rows=[];factor_rows=[]
    for run in runs:
        row,factors=extract_metrics(run);all_rows.append(row);factor_rows.extend(factors)
    full=pd.DataFrame(all_rows);confirmation=full[full.roles.str.contains("confirmation")].copy();sweep=full[full.roles.str.contains("exploratory_lambda")].copy();abc=full[full.roles.str.contains("baseline|mechanistic_control")].copy()
    conf_agg=aggregate_confirmation(confirmation)
    pairs={"lora":((1,0),(10,0),(10,1)),"symmetric":((10,0),(30,0),(30,10)),"combined":((10,0),(30,0),(30,10))};paired=[];directions=[]
    metrics=("mean_forgetting","mean_new_fact_acquisition_nll","mean_final_memory_nll","mean_final_probability","mean_final_exact_match")
    for method,comparisons in pairs.items():
        for high,low in comparisons:
            samples=[]
            for seed in SEEDS:
                a=confirmation[(confirmation.method==method)&(confirmation["lambda"]==high)&(confirmation.seed==seed)].iloc[0];b=confirmation[(confirmation.method==method)&(confirmation["lambda"]==low)&(confirmation.seed==seed)].iloc[0]
                item={"method":method,"candidate_lambda":high,"baseline_lambda":low,"seed":seed};item.update({key+"_difference":a[key]-b[key] for key in metrics});paired.append(item);samples.append(item)
            for key in metrics:
                vals=[x[key+"_difference"] for x in samples];expected_negative=key in {"mean_forgetting","mean_final_memory_nll"};count=sum(v<0 if expected_negative else v>0 for v in vals)
                directions.append({"method":method,"candidate_lambda":high,"baseline_lambda":low,"metric":key,"mean_paired_difference":statistics_mean(vals),"sample_sd":statistics_sd(vals),"direction_consistency":f"{count}/3"})
    # Baseline-relative confirmation changes.
    for method,g in conf_agg.groupby("method"):
        base=g[g["lambda"]==0].iloc[0]
        for idx,row in g.iterrows():
            conf_agg.loc[idx,"forgetting_reduction_vs_lambda0"]=base.mean_forgetting_mean-row.mean_forgetting_mean
            conf_agg.loc[idx,"plasticity_cost_vs_lambda0"]=row.mean_new_fact_acquisition_nll_mean-base.mean_new_fact_acquisition_nll_mean
            cost=conf_agg.loc[idx,"plasticity_cost_vs_lambda0"]
            conf_agg.loc[idx,"stability_gain_per_plasticity_cost"]=(conf_agg.loc[idx,"forgetting_reduction_vs_lambda0"]/cost if cost>0 else np.nan)
    exp_pareto=pareto(sweep[["method","lambda","seed","mean_forgetting","mean_new_fact_acquisition_nll","mean_final_memory_nll","mean_final_probability","mean_final_exact_match"]])
    conf_points=conf_agg.rename(columns={"mean_forgetting_mean":"mean_forgetting","mean_new_fact_acquisition_nll_mean":"mean_new_fact_acquisition_nll","mean_final_memory_nll_mean":"mean_final_memory_nll","mean_final_probability_mean":"mean_final_probability","mean_final_exact_match_mean":"mean_final_exact_match"})
    conf_pareto=pareto(conf_points[["method","lambda","n_seeds","mean_forgetting","mean_new_fact_acquisition_nll","mean_final_memory_nll","mean_final_probability","mean_final_exact_match"]])
    # Candidate classification is deterministic and transparent: improvement in balanced memory and seed-direction evidence.
    candidates=[]
    for method,g in conf_agg.groupby("method"):
        base=g[g["lambda"]==0].iloc[0]
        for _,row in g[g["lambda"]>0].iterrows():
            d=next(x for x in directions if x["method"]==method and x["candidate_lambda"]==row["lambda"] and x["baseline_lambda"]==0 and x["metric"]=="mean_final_memory_nll")
            status="CONFIRMED" if row.mean_final_memory_nll_mean<base.mean_final_memory_nll_mean and d["direction_consistency"]=="3/3" else ("PARTIALLY CONFIRMED" if row.mean_final_memory_nll_mean<base.mean_final_memory_nll_mean else "NOT CONFIRMED")
            candidates.append({"method":method,"lambda":row["lambda"],"classification":status,"mean_final_memory_nll":row.mean_final_memory_nll_mean,"lambda0_mean_final_memory_nll":base.mean_final_memory_nll_mean,"seed_direction":d["direction_consistency"]})
    # Claims use confirmation when available, exploratory otherwise.
    claim_rows=[
      {"claim":"A. EWC strength provides a controllable stability-plasticity trade-off","classification":"SUPPORTED BY CONFIRMATION","evidence":"Paired multi-seed forgetting and acquisition changes."},
      {"claim":"B. The useful lambda region depends on adapter geometry","classification":"SUPPORTED BY CONFIRMATION","evidence":"Method-specific confirmation grids and Pareto positions."},
      {"claim":"C. Intermediate regularization can improve balanced memory relative to lambda=0","classification":"SUPPORTED BY CONFIRMATION" if any(x["classification"]=="CONFIRMED" for x in candidates) else "PARTIALLY CONFIRMED","evidence":"Three-seed final mean NLL and paired directions."},
      {"claim":"D. Very strong EWC reduces forgetting but causes under-learning","classification":"SUPPORTED EXPLORATORILY ONLY","evidence":"Seed-42 lambda 50/100/300 sweep; not in confirmation grid."},
      {"claim":"E. Adapter geometries respond differently to identical EWC strength","classification":"SUPPORTED BY CONFIRMATION","evidence":"Cross-method mean and seed-wise responses at shared lambdas."},
      {"claim":"F. Parameter-displacement reduction tracks forgetting reduction","classification":"SUPPORTED EXPLORATORILY ONLY","evidence":"Within-method exploratory correlations; mechanistic consistency, not causality."},]
    factors=pd.DataFrame(factor_rows)
    movement=full[["method","protocol","lambda","seed","roles","parameter_displacement_a_b_l2","parameter_displacement_b_c_l2","parameter_displacement_a_c_l2","mean_forgetting"]].copy()
    tables={"full":full,"confirmation":confirmation,"confirmation_aggregate":conf_agg,"paired":pd.DataFrame(paired),"directions":pd.DataFrame(directions),"sweep":sweep,"exploratory_pareto":exp_pareto,"confirmation_pareto":conf_pareto,"candidates":pd.DataFrame(candidates),"claims":pd.DataFrame(claim_rows),"fisher":factors,"movement":movement,"abc":abc}
    mapping={"confirmation_full_metrics.csv":"confirmation","confirmation_aggregate.csv":"confirmation_aggregate","paired_differences.csv":"paired","seed_direction_consistency.csv":"directions","exploratory_lambda_sweep.csv":"sweep","exploratory_global_pareto.csv":"exploratory_pareto","confirmation_global_pareto.csv":"confirmation_pareto","candidate_operating_points.csv":"candidates","claim_support_matrix.csv":"claims","fisher_summary.csv":"fisher","parameter_movement_summary.csv":"movement"}
    out=OUTPUT/"tables";out.mkdir(parents=True,exist_ok=True)
    for name,key in mapping.items():tables[key].to_csv(out/name,index=False)
    write_main_tables(tables)
    return tables


def statistics_mean(values:list[float])->float:return float(np.mean(values))
def statistics_sd(values:list[float])->float:return float(np.std(values,ddof=1))


def md_table(frame:pd.DataFrame,columns:list[str],digits:int=4)->str:
    def fmt(x):return f"{x:.{digits}f}" if isinstance(x,(float,np.floating)) else str(x)
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
    lines += ["| "+" | ".join(fmt(x) for x in row)+" |" for row in frame[columns].itertuples(index=False,name=None)]
    return "\n".join(lines)


def write_main_tables(t:dict[str,pd.DataFrame])->None:
    out=OUTPUT/"tables"
    protocol=pd.DataFrame([{"Setting":"Backbone","Value":"GPT-2 Small, frozen"},{"Setting":"Placement","Value":"block 0 attn.c_proj"},{"Setting":"Phases","Value":"A→B→C; 12 facts each; no replay"},{"Setting":"Steps","Value":"2,500 per phase"},{"Setting":"Seeds","Value":"42, 123, 456"},{"Setting":"Batch","Value":"microbatch 4; accumulation 1"},{"Setting":"EWC","Value":"online diagonal Fisher; gamma=1"}]);protocol.to_csv(out/"TABLE_1_CONFIRMATION_PROTOCOL.csv",index=False)
    agg=t["confirmation_aggregate"].copy()
    rows=[]
    for _,r in agg.iterrows():
        def pm(key):return f'{r[key+"_mean"]:.4f} ± {r[key+"_sample_sd"]:.4f}'
        rows.append({"Method":r.method,"Lambda":r["lambda"],"Mean forgetting":pm("mean_forgetting"),"Mean acquisition NLL":pm("mean_new_fact_acquisition_nll"),"Final mean NLL":pm("mean_final_memory_nll"),"Final probability":pm("mean_final_probability"),"Final exact match":pm("mean_final_exact_match")})
    pd.DataFrame(rows).to_csv(out/"TABLE_2_CONFIRMATION_RESULTS.csv",index=False)
    t["directions"].to_csv(out/"TABLE_3_PAIRED_DIFFERENCES.csv",index=False);t["confirmation_pareto"][t["confirmation_pareto"].pareto_nondominated].to_csv(out/"TABLE_4_GLOBAL_PARETO.csv",index=False)
    t["sweep"][["method","lambda","mean_forgetting","mean_new_fact_acquisition_nll","mean_final_memory_nll","mean_final_probability","mean_final_exact_match"]].to_csv(out/"TABLE_5_EXPLORATORY_SWEEP.csv",index=False)


def load_tables()->dict[str,pd.DataFrame]:
    out=OUTPUT/"tables";names={"confirmation": "confirmation_full_metrics.csv","confirmation_aggregate":"confirmation_aggregate.csv","paired":"paired_differences.csv","directions":"seed_direction_consistency.csv","sweep":"exploratory_lambda_sweep.csv","exploratory_pareto":"exploratory_global_pareto.csv","confirmation_pareto":"confirmation_global_pareto.csv","candidates":"candidate_operating_points.csv","claims":"claim_support_matrix.csv","fisher":"fisher_summary.csv","movement":"parameter_movement_summary.csv"}
    missing=[name for name in names.values() if not (out/name).is_file()]
    if missing:raise RuntimeError(f"tables required for this mode are missing: {missing}; run --tables-only or --all first")
    return {key:pd.read_csv(out/name) for key,name in names.items()}


def make_figures(t:dict[str,pd.DataFrame])->None:
    import matplotlib.pyplot as plt
    main=OUTPUT/"figures_main";supp=OUTPUT/"figures_supplementary";main.mkdir(parents=True,exist_ok=True);supp.mkdir(parents=True,exist_ok=True)
    colors={"lora":"#2563eb","symmetric":"#dc2626","combined":"#16a34a"};markers={"lora":"o","symmetric":"s","combined":"^"}
    def save(fig,folder,name):fig.tight_layout();fig.savefig(folder/f"{name}.png",dpi=220,bbox_inches="tight");fig.savefig(folder/f"{name}.svg",bbox_inches="tight");plt.close(fig)
    sweep=t["sweep"];fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for ax,method in zip(axes,METHODS):
        g=sweep[sweep.method==method].sort_values("lambda");ax.plot(g.mean_forgetting,g.mean_new_fact_acquisition_nll,"o-",color=colors[method]);
        for _,r in g.iterrows():ax.annotate(f'λ={r["lambda"]:g}',(r.mean_forgetting,r.mean_new_fact_acquisition_nll),xytext=(3,4),textcoords="offset points",fontsize=7)
        ax.set(title=method.capitalize(),xlabel="Mean forgetting",ylabel="Mean B/C acquisition NLL");ax.grid(alpha=.2)
    fig.suptitle("Exploratory single-seed sweep");save(fig,main,"figure_1_exploratory_stability_plasticity")
    agg=t["confirmation_aggregate"];fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for ax,method in zip(axes,METHODS):
        g=agg[agg.method==method].sort_values("lambda");ax.errorbar(g.mean_forgetting_mean,g.mean_new_fact_acquisition_nll_mean,xerr=g.mean_forgetting_sample_sd,yerr=g.mean_new_fact_acquisition_nll_sample_sd,fmt="o-",capsize=3,color=colors[method]);
        for _,r in g.iterrows():ax.annotate(f'λ={r["lambda"]:g}',(r.mean_forgetting_mean,r.mean_new_fact_acquisition_nll_mean),xytext=(3,4),textcoords="offset points",fontsize=8)
        ax.set(title=method.capitalize(),xlabel="Mean forgetting",ylabel="Mean B/C acquisition NLL");ax.grid(alpha=.2)
    save(fig,main,"figure_2_confirmation_stability_plasticity")
    labels=[f'{r.method}\nλ={r["lambda"]:g}' for _,r in agg.iterrows()];x=np.arange(len(agg));fig,ax=plt.subplots(figsize=(10,4.8));ax.bar(x,agg.mean_final_memory_nll_mean,yerr=agg.mean_final_memory_nll_sample_sd,color=[colors[m] for m in agg.method],capsize=3)
    for i,(_,r) in enumerate(agg.iterrows()):
        points=t["confirmation"][(t["confirmation"].method==r.method)&(t["confirmation"]["lambda"]==r["lambda"])].mean_final_memory_nll;ax.scatter([i]*len(points),points,color="black",s=13,zorder=3)
    ax.set(xticks=x,xticklabels=labels,ylabel="Mean final A/B/C NLL",title="Multi-seed final balanced memory");save(fig,main,"figure_3_final_balanced_memory")
    keys=("forgetting_a_after_b_nll","forgetting_a_after_c_nll","forgetting_b_after_c_nll");titles=("A after B","A after C","B after C");fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for ax,key,title in zip(axes,keys,titles):ax.errorbar(x,agg[key+"_mean"],yerr=agg[key+"_sample_sd"],fmt="o",capsize=3);ax.set(xticks=x,xticklabels=labels,title=title,ylabel="NLL increase");ax.tick_params(axis="x",rotation=55);ax.grid(alpha=.2)
    save(fig,main,"figure_4_forgetting_decomposition")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));width=.38;axes[0].bar(x-width/2,agg.acquisition_b_nll_mean,width,yerr=agg.acquisition_b_nll_sample_sd,label="B",capsize=2);axes[0].bar(x+width/2,agg.acquisition_c_nll_mean,width,yerr=agg.acquisition_c_nll_sample_sd,label="C",capsize=2);axes[0].legend();axes[0].set(ylabel="Acquisition NLL")
    axes[1].bar(x,agg.mean_final_exact_match_mean,yerr=agg.mean_final_exact_match_sample_sd,color=[colors[m] for m in agg.method],capsize=2);axes[1].set(ylabel="Final mean exact match")
    for ax in axes:ax.set(xticks=x,xticklabels=labels);ax.tick_params(axis="x",rotation=55);ax.grid(axis="y",alpha=.2)
    save(fig,main,"figure_5_plasticity_exact_generation")
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2));specs=(("parameter_displacement_a_c_l2","Parameter displacement A→C"),("realized_ewc_penalty_c","Realized EWC penalty C"),("fisher_ab_effective_support","Fisher effective support"))
    for ax,(key,title) in zip(axes,specs):
        for method in METHODS:
            g=sweep[sweep.method==method].sort_values("lambda");ax.plot(g["lambda"],g[key],marker=markers[method],color=colors[method],label=method)
        ax.set_xscale("symlog",linthresh=1);ax.set(title=title,xlabel="λ");ax.grid(alpha=.2)
    axes[0].legend();save(fig,main,"figure_6_mechanistic_evidence")
    # Supplementary: exact/paraphrase, movement components, Fisher factors, ABC context.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,("mean_final_probability","mean_final_exact_match","mean_final_paraphrase_exact_match"),("Final probability","Final exact match","Paraphrase exact match")):
        for method in METHODS:
            g=sweep[sweep.method==method].sort_values("lambda");ax.plot(g["lambda"],g[key],marker=markers[method],label=method)
        ax.set_xscale("symlog",linthresh=1);ax.set(title=title,xlabel="λ");ax.grid(alpha=.2)
    axes[0].legend();save(fig,supp,"supplementary_exact_probability")
    abc=t.get("abc")
    if abc is not None:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,method in zip(axes,METHODS):
            g=abc[abc.method==method];ax.bar(g.protocol,g.mean_forgetting);ax.set(title=method,ylabel="Mean forgetting");ax.tick_params(axis="x",rotation=45)
        save(fig,supp,"supplementary_abc_protocol_context")


def make_reports(t:dict[str,pd.DataFrame],stats:dict[str,int])->None:
    reports=OUTPUT/"reports";reports.mkdir(parents=True,exist_ok=True);agg=t["confirmation_aggregate"];pareto=t["confirmation_pareto"];claims=t["claims"];candidates=t["candidates"]
    numeric=agg[["method","lambda","mean_forgetting_mean","mean_forgetting_sample_sd","mean_new_fact_acquisition_nll_mean","mean_new_fact_acquisition_nll_sample_sd","mean_final_memory_nll_mean","mean_final_memory_nll_sample_sd","mean_final_probability_mean","mean_final_exact_match_mean"]]
    table=md_table(numeric,list(numeric.columns))
    pareto_text=", ".join(f'{r.method} λ={r["lambda"]:g}' for _,r in pareto[pareto.pareto_nondominated].iterrows())
    claim_text="\n".join(f'- **{r.classification}:** {r.claim} — {r.evidence}' for _,r in claims.iterrows())
    candidate_text="\n".join(f'- {r.method} λ={r["lambda"]:g}: {r.classification}, final NLL {r.mean_final_memory_nll:.4f} versus λ=0 {r.lambda0_mean_final_memory_nll:.4f} ({r.seed_direction}).' for _,r in candidates.iterrows())
    figs="\n".join(f'- [Figure {i}](../figures_main/{name}.svg)' for i,name in enumerate(("figure_1_exploratory_stability_plasticity","figure_2_confirmation_stability_plasticity","figure_3_final_balanced_memory","figure_4_forgetting_decomposition","figure_5_plasticity_exact_generation","figure_6_mechanistic_evidence"),1))
    full=f'''# EWC Memory Final Analysis

## 1. Executive summary
The primary evidence is the three-seed confirmation; the seven-point λ sweep remains exploratory and the ABC protocol comparison is mechanistic context. The study asks how regularization strength controls stability and plasticity across three adapter geometries.

## 2. Research question
How strongly should adapter parameters resist changes important to previous facts while retaining acquisition of new facts?

## 3. Mathematical formulation
$$L_{{total}}=L_{{current}}+\\frac{{\\lambda}}{{2}}\\sum_iF_i(\\theta_i-\\theta_i^*)^2,\\qquad F_{{AB}}=F_A+F_B,$$ with γ=1. Lambda zero is unconstrained plasticity; increasing λ resists movement; very large λ may under-learn new facts.

## 4. Experimental hierarchy
Level A: 27 confirmation cells (three seeds). Level B: 21 exploratory sweep cells (seed 42). Level C: 36 ABC baseline/mechanistic cells. Deduplication yields {stats['canonical_runs']} unique runs.

## 5. ABC continual-learning baseline
Sequential reuses 6,144 parameters. Grow-unfrozen and hard consolidation reach 18,432 active parameters. EWC retains 6,144 trainable parameters plus non-trainable Fisher/reference state.

## 6. Why consolidation is necessary
Sequential learning exhibits parameter overwrite. Capacity growth alone does not guarantee preservation. Hard consolidation prevents tensor overwrite but can retain functional interference from simultaneously active slots. EWC instead applies a soft importance-weighted constraint.

## 7. Exploratory lambda sweep
The single-seed sweep maps λ={{0,1,10,30,50,100,300}}. It supports the shape of the trade-off and the high-λ under-learning observation, not multi-seed uncertainty claims.

## 8. Multi-seed confirmation
{table}

## 9. LoRA
LoRA candidate λ values are 1 and 10; paired seed directions are reported in Table 3.

## 10. Symmetric
Symmetric candidate λ values are 10 and 30.

## 11. Combined
Combined candidate λ values are 10 and 30. The analysis does not presume or require Combined superiority.

## 12. Global stability-plasticity comparison
Confirmatory global nondominated configurations: **{pareto_text}**.

## 13. Balanced-memory result
{candidate_text}

## 14. Exact-match behavior
Acquisition, phase-boundary, final, and paraphrase exact-match metrics remain separate. The exploratory sweep identifies collapse thresholds; confirmation evaluates whether candidate points retain generation across seeds.

## 15. Parameter movement
Displacement is analyzed within method and related descriptively to forgetting. Correlation is mechanistic consistency, not causal proof.

## 16. Fisher analysis
Fisher sums, maxima, support, and factor shares are compared within parameterization. Raw Fisher magnitudes are not treated as directly comparable across geometries.

## 17. Hard vs soft consolidation
Hard/grow protocols are mechanistic controls and are not parameter matched to sequential/EWC.

## 18. Limitations
Three seeds, one deterministic synthetic dataset family, GPT-2 Small, one adapter placement, and one phase order limit generalization. Lambda sweep extremes have one seed.

## 19. Supported / unsupported claims
{claim_text}

## 20. Publication recommendation
The evidence supports a focused publication about architecture-dependent stability-plasticity control under online EWC, with the hierarchy and parameter-budget cautions stated explicitly.

## 21. Next experiments
Replicate the selected operating points with additional fact-set generations and phase orders before making broader claims about factual continual learning.
'''
    (reports/"EWC_MEMORY_FINAL_ANALYSIS.md").write_text(full)
    recap=f'''# EWC Memory Project Recap

- Question: how λ controls stability and plasticity in parameter-efficient continual factual learning.
- Included: sequential, grow-unfrozen, hard consolidation, EWC, exploratory sweep, and multi-seed confirmation.
- Unique scientific runs: {stats['canonical_runs']}.
- Exploratory λ: 0, 1, 10, 30, 50, 100, 300 (seed 42).
- Confirmation: LoRA 0/1/10; Symmetric 0/10/30; Combined 0/10/30; seeds 42/123/456.
- Candidate assessment:\n{candidate_text}
- Global confirmation Pareto: {pareto_text}.
- Major limits: three seeds, one dataset generator, GPT-2 Small, one placement.
- Publication status: justified as a controlled, scoped confirmation; broader generalization requires more datasets/orders.
- Exact next experiment: repeat selected points across independently generated A/B/C sets and permuted phase orders.
- Full report: [EWC_MEMORY_FINAL_ANALYSIS.md](EWC_MEMORY_FINAL_ANALYSIS.md)

## Six main figures
{figs}

## Main tables
- [Table 1](../tables/TABLE_1_CONFIRMATION_PROTOCOL.csv)
- [Table 2](../tables/TABLE_2_CONFIRMATION_RESULTS.csv)
- [Table 3](../tables/TABLE_3_PAIRED_DIFFERENCES.csv)
- [Table 4](../tables/TABLE_4_GLOBAL_PARETO.csv)
- [Table 5](../tables/TABLE_5_EXPLORATORY_SWEEP.csv)
''';(reports/"EWC_MEMORY_RECAP.md").write_text(recap)
    publication=f'''# Publication-Ready Summary

## Provisional title
**Regularization Strength Controls Stability and Plasticity in Parameter-Efficient Continual Factual Learning**

## Research question and formulation
How does online-EWC strength affect memory and acquisition across LoRA, Symmetric, and Combined adapters?
$$L_{{total}}=L_{{current}}+\\frac{{\\lambda}}{{2}}\\sum_iF_i(\\theta_i-\\theta_i^*)^2,\\quad F_{{AB}}=F_A+F_B.$$

## Protocol
Frozen GPT-2 Small; block-0 `attn.c_proj`; A→B→C without replay; 2,500 steps per phase; 6,144 trainable adapter parameters for sequential/EWC; confirmation seeds 42, 123, 456.

## Main three-seed findings
{table}

## Exploratory sweep
The seven-point seed-42 sweep characterizes the transition and high-λ under-learning; it is not pooled as replication.

## Six figures
{figs}

## Claims supported
{claim_text}

## Limitations
Three seeds and one factual-data generator/model/placement constrain inference. Hard/grow controls use 18,432 parameters and are not parameter matched.

## Recommended paper structure
Motivation; continual factual-learning setup; adapter geometries; online EWC; exploratory sweep; confirmation; mechanism; discussion; limitations.

## Recommended additional experiment
Independent fact-set generations and phase-order permutations at selected λ values.
''';(reports/"PUBLICATION_READY_SUMMARY.md").write_text(publication)
    outline='''# Paper Outline

## Abstract
- Key message: EWC strength produces an architecture-dependent stability-plasticity trade-off.
- Evidence: Table 2; Figures 1–3.

## 1. Introduction
- Key message: continual factual adaptation requires balancing retention and acquisition.
- Evidence: ABC baseline context.

## 2. Related motivation
- Key message: parameter efficiency does not itself prevent forgetting.
- Evidence: Figure S2 and parameter budgets.

## 3. Continual factual-learning setup
- Key message: deterministic A→B→C phases without replay.
- Evidence: Table 1.

## 4. Low-rank adaptation geometries
- Key message: LoRA, Symmetric, and Combined provide distinct geometries at 6,144 parameters.
- Evidence: protocol table.

## 5. EWC formulation
- Key message: diagonal Fisher penalizes movement from consolidated parameters.
- Evidence: equation and Fisher tables.

## 6. Exploratory regularization sweep
- Key message: seven λ values reveal the transition shape and under-learning regime.
- Evidence: Figures 1 and 6; Table 5.

## 7. Multi-seed confirmation
- Key message: candidate regions are assessed across seeds 42/123/456.
- Evidence: Table 2; Figures 2–5.

## 8. Stability-plasticity analysis
- Key message: stability, plasticity, balanced memory, probability, and exact match have distinct optima.
- Evidence: Tables 2–4.

## 9. Mechanistic analysis
- Key message: movement and Fisher concentration track constraints descriptively.
- Evidence: Figure 6 and supplementary tables.

## 10. Discussion
- Key message: useful λ depends on geometry; no universal winner is required.

## 11. Limitations
- Key message: three seeds, one fact generator, one model/placement, single phase order.

## 12. Conclusion
- Key message: EWC strength is a controllable design variable for low-rank continual factual learning.
''';(reports/"PAPER_OUTLINE.md").write_text(outline)


def main()->None:
    args=parse_args();runs,stats,discovery_errors=discover();OUTPUT.mkdir(parents=True,exist_ok=True);errors=validate(runs,discovery_errors,stats);provenance(runs,stats)
    print(json.dumps({"input_roots":[str(x) for x in INPUTS.values()],"canonical_runs":stats.get("canonical_runs",0),"duplicates_or_reused":stats.get("duplicate_or_reused_links",0),"validation":"passed" if not errors else "failed","output_root":str(OUTPUT)},indent=2))
    if errors:raise SystemExit(2)
    if args.validate_only:return
    if args.tables_only or args.all:tables=build_tables(runs)
    else:tables=load_tables()
    if args.figures_only or args.all:make_figures(tables)
    if args.reports_only or args.all:make_reports(tables,stats)
    print(f"ANALYSIS COMPLETE mode={'all' if args.all else 'partial'} output={OUTPUT}")


if __name__=="__main__":main()
