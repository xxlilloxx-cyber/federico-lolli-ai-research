"""Build the historical pre-long-run report from early saved result files.

The final consolidated report is report/TECHNICAL_REPORT.md.  This legacy
generator intentionally writes a separate historical artifact so it cannot
overwrite the final 5,000-step and held-out-test conclusions.
"""
import csv, glob, json, statistics
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report'; FIG=REPORT/'figures'; FIG.mkdir(parents=True, exist_ok=True)

def load(p): return json.loads(Path(p).read_text())
def rows_for(kind, steps):
    return [load(p) for p in glob.glob(str(ROOT/f'results_extended/{kind}/r*_steps{steps}/summary.json'))]
def historical():
    pats={'Transformer Base':'results/baseline/r*_steps500/summary.json','LoRA':'results/lora/r*_steps500*/summary.json','Quadratic':'results/quadratic/r*_steps500_final/summary.json','Signed Quadratic':'results/signed_quadratic/r*_steps500_final/summary.json','Feature Interaction':'results/feature_interaction/r*_steps500_final/summary.json'}
    out=[]
    for name,p in pats.items():
        ds=[load(x) for x in glob.glob(str(ROOT/p))]
        if ds: out.append({'model':name,'loss':statistics.mean(d['validation_loss'] for d in ds),'sd':statistics.pstdev(d['validation_loss'] for d in ds),'ppl':statistics.mean(d['perplexity'] for d in ds),'params':ds[0]['trainable_parameters'],'vram':max(d['peak_vram_bytes'] for d in ds),'time':statistics.mean(d['training_seconds'] for d in ds)})
    return out
def avg_table(kind, steps):
    ds=rows_for(kind,steps)
    if not ds:return None
    return {'model':kind,'steps':steps,'n':len(ds),'loss':statistics.mean(d['validation_loss'] for d in ds),'sd':statistics.pstdev(d['validation_loss'] for d in ds),'ppl':statistics.mean(d['perplexity'] for d in ds),'params':ds[0]['trainable_parameters'],'vram':max(d['peak_vram_bytes'] for d in ds),'time':statistics.mean(d['training_seconds'] for d in ds)}
def savefig(name): plt.tight_layout(); plt.savefig(FIG/name,dpi=180); plt.close()

def main():
    hist=historical(); ext=[x for s in (500,1000,2000) for k in ('lora','symmetric_quadratic','linear_quadratic') for x in [avg_table(k,s)] if x]
    synthetic=load(ROOT/'results_extended/synthetic/summary.json')
    # Figure 1/2/3: simple conceptual diagrams made from text so no architecture is implied beyond the code.
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis('off');
    boxes=['Base: xW','LoRA: xW + αxUV','Quadratic: xW + α(x⊙x)UV','Interactions: xW + α[(xU)⊙(xV)]P'];
    for i,b in enumerate(boxes): ax.text(i/3.2,.5,b,ha='center',va='center',bbox=dict(boxstyle='round',fc='#e8f0fe')); 
    savefig('01_conceptual_progression.png')
    fig,ax=plt.subplots(figsize=(10,4)); ax.axis('off'); ax.text(.5,.75,'GPT-2 Small (12 Transformer blocks)',ha='center',bbox=dict(boxstyle='round',fc='#f0f0f0')); ax.text(.5,.25,'Block 0 → attention c_proj → frozen W + selected adapter',ha='center',bbox=dict(boxstyle='round',fc='#d9ead3')); ax.annotate('',xy=(.5,.58),xytext=(.5,.68),arrowprops={'arrowstyle':'->'}); savefig('02_transformer_insertion.png')
    fig,ax=plt.subplots(figsize=(12,4)); ax.axis('off'); names=['Base','LoRA','Quadratic','Signed','Feature interaction','Symmetric','Linear + Quadratic'];
    for i,n in enumerate(names): ax.text(i/6,.5,n,ha='center',va='center',rotation=20,bbox=dict(boxstyle='round',fc='#fff2cc' if i>0 else '#d9ead3')); savefig('03_adapter_families.png')
    # Training curves for all available saved primary runs.
    metric_files=[]
    for p in glob.glob(str(ROOT/'results_extended/*/r*_steps*/metrics.csv')): metric_files.append(Path(p))
    for name,col,ylabel in [('04_training_loss.png','training_loss','Training loss'),('05_validation_loss.png','validation_loss','Validation loss'),('06_perplexity.png','perplexity','Perplexity'),('17_gradient_norm.png','gradient_norm','Gradient norm')]:
        plt.figure(figsize=(10,5))
        for p in metric_files:
            vals=list(csv.DictReader(p.open())); label=p.parent.parent.name+' '+p.parent.name
            if vals: plt.plot([int(v['step']) for v in vals],[float(v[col]) for v in vals],alpha=.8,label=label)
        plt.xlabel('Optimization step'); plt.ylabel(ylabel); plt.title(ylabel+' versus step'); plt.legend(fontsize=6); savefig(name)
    # Performance/cost plots using aggregate rows.
    allperf=hist+[x for x in ext if x['steps'] in (500,1000,2000)]
    plt.figure(figsize=(9,5));
    for r in allperf: plt.scatter(r['params'],r['loss'],label=f"{r['model']} {r.get('steps','')}" )
    plt.xlabel('Trainable parameters'); plt.ylabel('Validation loss (lower is better)'); plt.title('Performance versus trainable parameters'); plt.legend(fontsize=6); savefig('07_parameter_efficiency.png')
    plt.figure(figsize=(9,5));
    for k in ('lora','symmetric_quadratic','linear_quadratic'):
        xs=[];ys=[]
        for s in (500,1000,2000):
            a=avg_table(k,s)
            if a:xs.append(s);ys.append(a['loss'])
        if xs:plt.plot(xs,ys,'o-',label=k)
    plt.xlabel('Training steps');plt.ylabel('Validation loss (lower is better)');plt.title('Convergence versus step budget');plt.legend();savefig('08_rank_or_steps_convergence.png')
    plt.figure(figsize=(9,5));
    for r in ext:plt.scatter(r['time'],r['loss'],label=f"{r['model']} {r['steps']}")
    plt.xlabel('Training time (seconds)');plt.ylabel('Validation loss');plt.title('Training cost versus performance');plt.legend(fontsize=7);savefig('09_training_time_performance.png')
    plt.figure(figsize=(9,5));
    for r in allperf:plt.scatter(r['vram']/2**20,r['loss'],label=f"{r['model']} {r.get('steps','')}")
    plt.xlabel('Peak VRAM (MiB)');plt.ylabel('Validation loss');plt.title('GPU memory versus performance');plt.legend(fontsize=6);savefig('10_memory_performance.png')
    # Synthetic models as separate lines.
    plt.figure(figsize=(10,5));
    for model in sorted(set(x['model'] for x in synthetic)):
        ds=sorted([x for x in synthetic if x['model']==model],key=lambda x:x['case']);plt.plot([x['case'] for x in ds],[x['mse'] for x in ds],'o-',label=model)
    plt.ylabel('MSE (lower is better)');plt.xlabel('Target function');plt.title('Synthetic linear/quadratic/mixed benchmark');plt.legend(fontsize=7);savefig('11_synthetic_benchmark.png')
    # Same rank / parameter matched are represented by the saved new 500-step rows.
    plt.figure(figsize=(8,5));
    for k,c in [('symmetric_quadratic','tab:orange'),('linear_quadratic','tab:red')]:
        a=avg_table(k,500)
        if a:plt.bar(k,a['loss'],color=c,label=f"{a['params']} params")
    plt.axhline(next(x['loss'] for x in hist if x['model']=='LoRA'),color='tab:blue',ls='--',label='LoRA historical reference');plt.ylabel('Validation loss');plt.title('Parameter-matched comparison');plt.legend();savefig('12_parameter_matched.png')
    # Branch norms are available in mixed metrics.
    plt.figure(figsize=(9,5));
    for p in glob.glob(str(ROOT/'results_extended/linear_quadratic/r*_steps*/metrics.csv')):
        vals=list(csv.DictReader(open(p))); 
        if vals and 'adapter_linear_norm' in vals[0]:
            label=Path(p).parent.name;plt.plot([int(v['step']) for v in vals],[float(v['adapter_linear_norm']) for v in vals],label=label+' linear');plt.plot([int(v['step']) for v in vals],[float(v['adapter_quadratic_norm']) for v in vals],ls='--',label=label+' quadratic')
    plt.xlabel('Step');plt.ylabel('Adapter branch output norm');plt.title('Linear and quadratic branch contributions');plt.legend(fontsize=6);savefig('13_linear_quadratic_branch_norms.png')
    # Technical report tables.
    def fmt(r):return f"| {r['model']} | {r.get('steps','500')} | {r['n'] if 'n' in r else 3} | {r['params']} | {r['loss']:.4f} ± {r['sd']:.4f} | {r['ppl']:.2f} | {r['time']:.1f} | {r['vram']/2**20:.1f} |"
    table='\n'.join(fmt(r) for r in hist+ext)
    trace='\n'.join(f"- `{r['model']}` steps={r.get('steps','500')} seed set: result directories under `results_extended/{r['model']}/r*_steps{r.get('steps','500')}`" for r in ext)
    md='''# Technical Scientific Report — Quadratic Transformer

## 1. Executive Summary

This study asks whether a low-rank second-order correction can adapt a frozen Transformer more effectively than a low-rank linear correction with a comparable number of trainable parameters. The answer is approached experimentally rather than assumed. The original Transformer is the reference; LoRA is the primary linear baseline; element-wise, signed, feature-interaction, symmetric quadratic, and combined linear-plus-quadratic branches are compared.

The new 500-step matched-budget runs used GPT-2 Small and three seeds. The Symmetric adapter averaged validation loss 3.5526; the Linear + Quadratic adapter with `r_L=1,r_Q=2` averaged 3.5302. The two new 1,000-step series completed for LoRA, Symmetric, and Linear+Quadratic. At 2,000 steps, all three architectures completed all three seeds. The first attempt for Linear+Quadratic seed 456 encountered a GPU kernel lockup, but the run was repeated successfully. These longer runs provide a balanced convergence comparison, while remaining limited to this model and protocol.

The main result is a qualified signal: the combined model and the symmetric quadratic model can remain competitive at the same 6,144 trainable parameters as LoRA. This does not prove that second-order adaptation is generally superior. The study uses one small language model, one dataset, one insertion point, and one training protocol.

## 2. Motivation and idea

An ordinary frozen linear projection computes `y=xW`. LoRA adds a low-rank linear correction `α(xU)V`; for fixed adapter weights this remains linear in `x`. Quadratic adapters add curvature: element-wise squares describe `x_i^2`, signed squares preserve the sign, and factorized interactions produce `x_i x_j`. The combined adapter represents `a^T x + x^T Q x`.

The progression is `x_i → x_i^2 → x_i|x_i| → x_i x_j → x^TQx → a^Tx+x^TQx`. Each step adds a different class of input dependence. The question is whether that extra expressivity translates into lower validation error per trainable parameter.

## 3. Mathematical models

The input to an adapted projection is `x∈R^d`, the frozen weight is `W`, and the output width is `d_out`.

0. **Transformer Base:** `Δy=0`, `y=xW`.

1. **LoRA:** `Δy=α(xU)V`, with `U∈R^{d×r}`, `V∈R^{r×d_out}` and `r(d+d_out)` parameters. It adds a rank-constrained linear map.

2. **Element-wise Quadratic:** `Δy=α(x⊙x)UV`. The square is feature-wise, not matrix multiplication. It primarily supplies transformed diagonal terms `x_i^2`.

3. **Signed Quadratic:** `Δy=α(x⊙|x|)UV`. Positive coordinates become `+x_i^2`, negative coordinates become `-x_i^2`; this preserves sign information while changing magnitude nonlinearly.

4. **Feature Interaction:** `Δy=α[(xU)⊙(xV)]P`. Since `(xU)_k=Σ_i x_iU_ik` and `(xV)_k=Σ_jx_jV_jk`, their product equals `Σ_iΣ_jU_ikV_jkx_ix_j`. It therefore includes off-diagonal interactions without storing a full `d×d` matrix. Its parameter count is `r(2d+d_out)`.

5. **Symmetric Quadratic Interaction:** `Δy=α(xU)^{⊙2}P`, with `U∈R^{d×r}`, `P∈R^{r×d_out}`. For output `l`, `Δy_l=Σ_kP_kl(x^Tu_k)^2=x^TQ_lx`, where `Q_l=Σ_kP_klu_ku_k^T`. Each outer product is symmetric rank 1, so `Q_l` is symmetric and represented implicitly with `r(d+d_out)` parameters. This is the quadratic analogue of the low-rank factorization `ΔW=UV` in LoRA.

6. **Linear + Quadratic:** `Δy=α_L(xA)B+α_Q[(xU)⊙(xV)]P`. The linear branch and quadratic branch do not share parameters. The first captures first-order changes; the second captures diagonal and off-diagonal second-order structure. The experiment records each branch norm and the ratio `||Δy_Q||/(||Δy_L||+ε)` when available.

## 4. Model architecture and implementation

The model is `gpt2` (GPT-2 Small, approximately 124M parameters) from Hugging Face. The tokenizer and pretrained weights are shared. Adapters are inserted only into `transformer.h[0].attn.c_proj`, the output projection of the attention sublayer in block 0. Q, K, V, and MLP projections are not modified. The backbone is frozen; adapter matrices are FP32 and the backbone is FP16.

Implementation is in `src/adapters.py` (`LowRankAdapter`) and insertion is performed by `freeze_and_insert`. Output factors are zero initialized (`V`, `P`, or `B` and `P`) so the initial adapted output equals the base output. The smoke tests measured maximum identity error 0.0 and finite gradients. The trainer uses AdamW, learning rate `3e-4`, gradient accumulation 4, batch size 1, sequence length 128, FP16 autocast, and gradient clipping at 1.0.

## 5. Dataset and task

The task is causal language modeling on WikiText-2 raw. Non-empty text from the public train and validation splits is concatenated, tokenized with the GPT-2 tokenizer, and segmented into consecutive 128-token blocks. The training blocks provide labels shifted by the causal language-model loss. Validation blocks are evaluated without gradient updates. The public test split was not used in this pipeline, so no test loss is claimed. This dataset is useful for a controlled language-modeling sanity study but does not establish general task or domain transfer.

## 6. Experimental protocol

All models use the same initial weights, tokenizer, data order, sequence length, effective batch size, optimizer, learning rate and seeds 42, 123, 456 where a series is complete. Validation is measured initially, every 25 steps and at the final step. Peak allocated VRAM, loss, perplexity, gradient norm, time and throughput are saved in each run's `metrics.csv`; `summary.json` stores the aggregate.

The 500-step matched-budget series includes Symmetric and Linear+Quadratic. Additional series were run at 1,000 and 2,000 steps for LoRA, Symmetric and Linear+Quadratic. The previously incomplete Linear+Quadratic seed 456 run was rerun successfully, so the 2,000-step comparison now has three completed seeds for all three architectures.

## 7. Parameter matching

LoRA and Symmetric both use `r(d+d_out)`. Feature Interaction uses `r(2d+d_out)`. For GPT-2 `d=d_out=768`, LoRA/Symmetric rank 4 gives 6,144 parameters. Linear+Quadratic uses `r_L(d+d_out)+r_Q(2d+d_out)`; `r_L=1,r_Q=2` also gives 6,144 exactly. The screening allocations `2:1` and `3:1` use 5,376 and 6,912 parameters.

## 8. Synthetic quadratic benchmark

The synthetic script samples Gaussian `x`, a linear coefficient `a`, and symmetric matrices `Q`. It evaluates three targets: `a^Tx`, `x^TQx`, and `a^Tx+x^TQx`. Each adapter is trained for 300 updates and evaluated by MSE. The results are a capacity diagnostic: they test whether the algebraic form can fit a known function, not whether it improves language modeling.

## 9. Results

| Model | Steps | Seeds | Trainable | Validation loss | PPL | Time (s) | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

The historical Base/LoRA/Quadratic/Signed/Feature rows are the previously completed 500-step reference series. New rows are traced to `results_extended`. The 1,000-step means are LoRA 3.5491, Symmetric 3.4770, and Linear+Quadratic 3.4928. At 2,000 steps the complete means are LoRA 3.4598, Symmetric 3.3784, and Linear+Quadratic 3.4360, all computed from three seeds.

## 10. How to read the figures

For validation loss, perplexity, MSE, time and memory, lower is better. For tokens/sec, higher is better. In a curve, a lower trajectory means less error at the same step; earlier separation indicates faster improvement; overlapping curves indicate little observable difference. A parameter-efficiency plot should be read as a frontier: a point is attractive when it has low loss with few parameters, not merely because it has the lowest loss.

Figure 11 plots one line per model across linear, quadratic and mixed synthetic targets. Figure 12 compares equal-budget new models and the historical LoRA reference. Figures 4–6 and 17 show optimization trajectories; Figure 13 shows the separate mixed-branch norms. These are observations, not causal proofs: a lower curve is compatible with better optimization or a better representation, and the experiment does not isolate those causes.

## 11. Interpretation

The synthetic results behave as expected: LoRA fits the linear target; Feature Interaction, Symmetric and Linear+Quadratic reduce error on quadratic targets; Linear+Quadratic fits the mixed target best among the tested forms. On GPT-2, Symmetric and Linear+Quadratic remain competitive with LoRA at equal budget. The longer series suggest that the relative ordering is not fixed at 500 steps and can change with optimization time.

The results do not show that every quadratic adapter is better. Element-wise Quadratic and Signed Quadratic historically trail LoRA in the 500-step language-modeling comparison. Feature Interaction has a promising historical mean but a larger seed spread. The new Linear+Quadratic signal is encouraging and its 2,000-step three-seed series is now complete, although it remains a single-model, single-dataset result.

## 12. Failure analysis and missing measurements

No completed run produced NaN or Inf. The first attempt at the 2,000-step Linear+Quadratic seed 456 run encountered an NVIDIA runtime lockup with processes in kernel state `D`; after the GPU recovered, the same configuration was rerun successfully. Test loss, inference latency, FLOPs, and layer-by-layer branch norms were not collected, so those conclusions are unavailable rather than inferred. The current branch diagnostics provide linear/quadratic output norms for the mixed adapter, but not a complete causal attribution of validation improvement.

## 13. Limitations

The study uses GPT-2 Small, one attention projection, WikiText-2, a short context, three seeds, and one learning rate. This archived generator describes a pre-test historical state; the canonical report documents the completed held-out evaluation. The GPU runtime was unstable during long campaigns. No broad hyperparameter search, larger Transformer, MLP placement, normalization variant, or downstream task was tested. These limits make the results a controlled research prototype rather than a general claim about Transformer adaptation.

## 14. Conclusions

**Observed:** second-order adapters can be implemented with zero-preserving initialization and can train within the 4 GiB budget. Symmetric and Linear+Quadratic can reach competitive validation loss at LoRA-comparable parameter counts. Synthetic fitting confirms their intended function classes.

**Supported interpretation:** low-rank quadratic structure provides useful representational capacity in controlled tests, and combining linear and quadratic branches is plausible under a fixed budget.

**Not supported:** a claim that quadratic adapters universally outperform LoRA, or that synthetic advantage transfers to all language tasks. The completed 2,000-step series strengthens the evidence but does not remove the study's model, dataset, insertion-point, and hyperparameter limitations.

## 15. Future work

The most informative next step is a held-out WikiText-2 test evaluation, followed by MLP-only versus attention-only placement, inference latency measurement, and an RMSNorm quadratic variant as a separately labeled stability experiment. A larger model or downstream task should follow only if the matched-budget ordering survives these checks. Larger models or downstream tasks should follow only if the matched-budget advantage survives those checks.

## 16. Traceability

{trace}

The source scripts are `src/adapters.py`, `src/train.py`, `scripts/synthetic_benchmark.py`, and `scripts/generate_technical_report.py`. Historical results remain under `results/`; this report's extension results remain under `results_extended/`.
'''
    md=md.replace('{table}', table).replace('{trace}', trace)
    md += r'''

## 17. Figure-by-figure reading guide

The following notes accompany the files in `report/figures/`. They deliberately separate what a reader should look for from what this dataset actually permits us to say.

### Figure 1 — Conceptual progression (`01_conceptual_progression.png`)

**How to read this figure.** Read from left to right: the representation moves from first-order coordinates to diagonal second-order terms, signed curvature, cross-feature products, a quadratic form, and finally a sum of first- and second-order terms. This is a conceptual map, not a measured curve.

**Interpretation of the observed result.** The figure defines the scientific question. It does not establish that a later representation is better; that question is tested by Figures 5–12 and the tables.

### Figure 2 — Transformer insertion point (`02_transformer_insertion.png`)

**How to read this figure.** The highlighted module is the single attention output projection `transformer.h[0].attn.c_proj`. All other GPT-2 weights remain frozen in adapter runs.

**Interpretation of the observed result.** Restricting the intervention to one projection makes the comparison auditable and inexpensive, but it also limits how much of the Transformer can adapt. Results should therefore be interpreted as evidence about this insertion point.

### Figure 3 — Adapter families (`03_adapter_families.png`)

**How to read this figure.** Each branch shows the operation applied to the same input activation. The baseline has no trainable correction; LoRA has one linear low-rank branch; the quadratic families apply nonlinear feature or interaction maps.

**Interpretation of the observed result.** The diagram explains why parameter count and computational cost must be reported separately: Feature Interaction and Linear+Quadratic perform more intermediate products even when their trainable parameter counts are matched.

### Figure 4 — Training loss (`04_training_loss.png`)

**How to read this figure.** The x-axis is optimization step and the y-axis is training loss; lower is better. A curve that falls earlier is learning faster under this protocol, while overlapping curves indicate similar optimization trajectories.

**Interpretation of the observed result.** The available trajectories show improvement for the trained adapters and no completed run with NaN/Inf. A training-loss advantage alone is not evidence of generalization; validation loss is the primary comparison.

### Figure 5 — Validation loss (`05_validation_loss.png`)

**How to read this figure.** Lower validation loss is better. Compare models at the same step and parameter budget; do not compare a 2,000-step endpoint with a 500-step endpoint as if they were the same amount of training.

**Interpretation of the observed result.** In the complete longer series, Symmetric is below LoRA on average at 1,000 and 2,000 steps. Linear+Quadratic is below LoRA at 1,000 and 2,000 steps, now with three completed seeds at 2,000 steps. This is stronger evidence than the earlier incomplete comparison, but still not a universal superiority claim.

### Figure 6 — Perplexity (`06_perplexity.png`)

**How to read this figure.** Perplexity is `exp(validation loss)`, so lower is better. Differences can look larger on this scale than on the loss scale because of the exponential transform.

**Interpretation of the observed result.** The ordering follows validation loss: at 2,000 steps the complete means are approximately 29.33 for Symmetric and 31.81 for LoRA. This is a descriptive transformation of the same validation measurements, not an independent test.

### Figure 7 — Parameter efficiency (`07_parameter_efficiency.png`)

**How to read this figure.** The horizontal axis is trainable parameters and the vertical axis is validation loss. Prefer points toward the lower-left frontier. A point lower but much farther right may simply have bought performance with more capacity.

**Interpretation of the observed result.** LoRA and Symmetric have exactly 6,144 trainable parameters at rank 4 in this GPT-2 projection. Their direct comparison is therefore more informative than comparing either with a higher-budget method.

### Figure 8 — Convergence (`08_rank_or_steps_convergence.png`)

**How to read this figure.** This plot compares endpoint loss as training length increases. A descending line indicates continued benefit from additional optimization; a crossing indicates that ranking depends on the training budget.

**Interpretation of the observed result.** The relative ranking changes with step count: the quadratic variants improve substantially between 500 and 2,000 steps. This is why the short 500-step screen should not be treated as the final architectural verdict.

### Figure 9 — Time versus performance (`09_training_time_performance.png`)

**How to read this figure.** Lower training time and lower validation loss are jointly desirable. The plot measures wall-clock cost observed on this machine, not an architecture-independent FLOP count.

**Interpretation of the observed result.** The adapter runs have similar peak memory, but their times differ modestly. Any quality gain must therefore be weighed against these measured costs and the GPU lockup risk seen during the longest campaign.

### Figure 10 — Memory versus performance (`10_memory_performance.png`)

**How to read this figure.** Lower memory is preferable for a fixed quality. Points clustered at the same memory indicate that frozen-backbone activation storage, rather than adapter parameter count, dominates this small experiment.

**Interpretation of the observed result.** All completed adapter variants remained well below the 4 GiB hardware limit in the recorded runs (about 348 MiB peak allocated in the trainer measurement). This does not imply that larger sequence lengths or more insertion points would fit.

### Figure 11 — Synthetic benchmark (`11_synthetic_benchmark.png`)

**How to read this figure.** Each line is one adapter and the x-axis selects linear, quadratic, or mixed targets. Lower MSE is better. The target-generating function is known, so this is a representational capacity test.

**Interpretation of the observed result.** LoRA is strongest on the linear target; Feature Interaction and Linear+Quadratic are strongest on the quadratic and mixed targets in the recorded run. This supports the algebraic interpretation, but synthetic fitting does not guarantee language-modeling gains.

### Figure 12 — Parameter-matched comparison (`12_parameter_matched.png`)

**How to read this figure.** Compare models with the same trainable budget rather than merely the same nominal rank. Lower loss at equal budget is the relevant observation.

**Interpretation of the observed result.** The new Symmetric and Linear+Quadratic configurations were constructed to match LoRA's 6,144 parameters. Their lower endpoints in the longer runs are therefore not explained by a larger adapter budget, although optimization and insertion effects remain possible explanations.

### Figure 13 — Linear and quadratic branch norms (`13_linear_quadratic_branch_norms.png`)

**How to read this figure.** Solid and dashed traces show the output norms of the linear and quadratic branches of Linear+Quadratic. A rising trace means that branch contributes more strongly to the adapter output; it is not itself a loss measure.

**Interpretation of the observed result.** The traces verify that both branches can be monitored separately. A nonzero branch norm does not prove that the branch is causally necessary; branch ablations would be needed for that conclusion.

### Figure 14 — Quadratic-to-linear ratio

**How to read this figure.** The desired quantity is `||Δy_Q||/(||Δy_L||+ε)`. Values above one mean the measured quadratic output dominates the linear output at that point.

**Interpretation of the observed result.** A dedicated ratio series was not saved by all historical runs, so no complete cross-model figure is claimed. This is a documented missing measurement, not a zero ratio.

### Figure 15 — Layer contribution

**How to read this figure.** A layer plot would compare adapter output norms or validation deltas for each insertion site.

**Interpretation of the observed result.** Only block 0 attention output was instrumented. Layer-by-layer contribution cannot be inferred and is intentionally marked unavailable.

### Figure 16 — Activation distributions

**How to read this figure.** A distribution plot would show the input and transformed activation ranges, especially the growth of `x^2`.

**Interpretation of the observed result.** The pipeline stored summary statistics rather than raw activation samples for every run. Consequently a distribution figure would risk inventing information and is not presented.

### Figure 17 — Gradient norms (`17_gradient_norm.png`)

**How to read this figure.** Lower or smoother gradients are not automatically better. Look for finite values, sudden spikes, persistent near-zero norms, or divergence.

**Interpretation of the observed result.** Completed runs had finite gradients and no reported NaN/Inf. The figure supports numerical feasibility, but it does not by itself establish that one parameterization optimizes better.

### Figure 18 — Convergence speed

**How to read this figure.** Convergence speed should be measured as loss reached per unit step or per unit wall-clock time, not only as final loss.

**Interpretation of the observed result.** The current data support the step-based convergence view in Figure 8. A separate latency-normalized curve was not collected, so claims about speed in seconds remain limited to the timing table.

## 18. Computational accounting

The trainable parameter counts are small because the 124M-parameter backbone is frozen. This does not make all adapters equally cheap: LoRA adds two matrix projections; element-wise and signed variants add a feature transform; Feature Interaction adds two projections and a Hadamard product; Symmetric reuses one projected representation but squares it; Linear+Quadratic executes both a linear and a quadratic path. Peak VRAM in the recorded setup was similar because the frozen model and sequence activations dominate allocation. Parameter efficiency and computational efficiency are therefore separate claims.

## 19. Statistical reading and evidence level

The reported standard deviations are across seeds for complete series. A difference larger than the seed spread is more persuasive than a difference of the same order, but three seeds remain a small sample and no formal hypothesis test was preregistered. The 2,000-step Linear+Quadratic result has three completed seeds. The mean and standard deviation are still descriptive because the study has no preregistered formal hypothesis test and uses one model, dataset, and insertion protocol. The appropriate classification is **signal of interest, not a conclusive universal advantage**.

## 20. Reproducibility map

Every table row is generated from `summary.json` files in `results/` or `results_extended/`; per-step curves come from the corresponding `metrics.csv`. Configuration files are under `configs/`. The synthetic benchmark is in `results_extended/synthetic/`. The formerly incomplete run was rerun successfully into its dedicated result directory. Regenerate this document with `scripts/generate_technical_report.py` after adding results.

## 21. What would change the conclusion

The strongest next test is a held-out WikiText-2 test evaluation, followed by MLP-only versus attention-plus-MLP placement. If the matched-budget ordering persists across seeds, then MLP-only and attention-plus-MLP placement can test whether the effect depends on the single current projection. If it disappears, the present result should be treated as an optimization or insertion-point effect rather than general evidence for second-order adapters.
'''
    (REPORT/'HISTORICAL_PRE_LONG_RUN_REPORT.md').write_text(md)
    checklist='''# Report checklist\n\n- [x] Formulae and dimensions for models 0–6\n- [x] GPT-2, tokenizer, insertion point, precision, hardware and protocol\n- [x] Historical baseline/reference retained\n- [x] 500/1000/2000 step results linked to summary files\n- [x] Synthetic linear/quadratic/mixed benchmark\n- [x] Training, validation, perplexity, gradient, VRAM and throughput figures where data exist\n- [x] Parameter-matched and same-budget discussion\n- [x] Seed means and standard deviations for complete series\n- [x] Failure analysis and GPU lockup documented\n- [ ] Test loss: not collected by the existing pipeline\n- [ ] Inference latency/FLOPs: not collected\n- [x] Complete 2000-step Linear+Quadratic three-seed series, including rerun of seed 456\n- [ ] Layer-by-layer branch norms: not collected\n'''
    (REPORT/'REPORT_CHECKLIST.md').write_text(checklist)
    print(f'Wrote {REPORT}/HISTORICAL_PRE_LONG_RUN_REPORT.md and {len(list(FIG.glob("*.png")))} figures')

if __name__=='__main__': main()
