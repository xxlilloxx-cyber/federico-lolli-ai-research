import csv,json,math,statistics
from pathlib import Path
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; base=root/'results_long_convergence_5000'/'budget12288_5000'; testdir=root/'results_long_convergence_5000'/'test_eval'; out=root/'report'/'LONG_CONVERGENCE_AND_TEST'; fig=out/'figures'; out.mkdir(parents=True,exist_ok=True); fig.mkdir(exist_ok=True)
rows=[]
for p in sorted(base.glob('*/summary.json')):
 s=json.load(open(p)); m=list(csv.DictReader(open(p.parent/'metrics.csv'))); curve={int(r['step']):float(r['validation_loss']) for r in m}; test=json.load(open(testdir/(p.parent.name+'.json'))); rows.append({'name':p.parent.name,'strategy':s['strategy'],'model':s['kind'],'seed':int(s['seed']),'val':s['validation_loss'],'ppl':s['perplexity'],'time':s['training_seconds'],'tps':s['tokens_per_second'],'params':s['trainable_parameters'],'vram':s['peak_vram_bytes']/2**20,'curve':curve,'test':test['test_loss'],'test_ppl':test['test_perplexity']})
fields=['name','strategy','model','seed','val','ppl','test','test_ppl','time','tps','params','vram']
with open(out/'long_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
def z(st,mo): return [r for r in rows if r['strategy']==st and r['model']==mo]
def mean_sd(xs): return statistics.mean(xs),statistics.stdev(xs) if len(xs)>1 else 0
# Aggregate table
agg=[]
for st in ['concentrated','two_distributed','four_distributed']:
 for mo in ['lora','symmetric_quadratic']:
  zz=z(st,mo); vals=[r['val'] for r in zz]; tes=[r['test'] for r in zz]; aulc=[]
  for r in zz: aulc.append(sum(r['curve'].values())/len(r['curve']))
  agg.append({'strategy':st,'model':mo,'n':len(zz),'val_mean':statistics.mean(vals),'val_sd':statistics.stdev(vals),'ppl_mean':statistics.mean([r['ppl'] for r in zz]),'ppl_sd':statistics.stdev([r['ppl'] for r in zz]),'test_mean':statistics.mean(tes),'test_sd':statistics.stdev(tes),'test_ppl_mean':statistics.mean([r['test_ppl'] for r in zz]),'test_ppl_sd':statistics.stdev([r['test_ppl'] for r in zz]),'gap_mean':statistics.mean([r['test']-r['val'] for r in zz]),'gap_sd':statistics.stdev([r['test']-r['val'] for r in zz]),'aulc_mean':statistics.mean(aulc),'time_mean':statistics.mean([r['time'] for r in zz]),'tps_mean':statistics.mean([r['tps'] for r in zz]),'vram_mean':statistics.mean([r['vram'] for r in zz])})
with open(out/'aggregate_long.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
# checkpoint table and delta
cp=[1000,2000,3000,4000,5000]; checkpoint=[]
for st in ['concentrated','two_distributed','four_distributed']:
 for mo in ['lora','symmetric_quadratic']:
  zz=z(st,mo)
  for step in cp:
   vv=[r['curve'].get(step) for r in zz]; vv=[x for x in vv if x is not None]; checkpoint.append({'strategy':st,'model':mo,'step':step,'mean':statistics.mean(vv),'sd':statistics.stdev(vv) if len(vv)>1 else 0})
with open(out/'checkpoint_means.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=checkpoint[0].keys());w.writeheader();w.writerows(checkpoint)
# plots helper
colors={'lora':'tab:blue','symmetric_quadratic':'tab:orange'}; ls={'concentrated':'-','two_distributed':'--','four_distributed':':'}
for st,title,name in [('four_distributed','5000-step validation convergence — four-layer','validation_four_layer_5000.png'),('two_distributed','5000-step validation convergence — two-layer','validation_two_layer_5000.png'),('concentrated','5000-step validation convergence — concentrated','validation_concentrated_5000.png')]:
 plt.figure(figsize=(8,5));
 for mo in colors:
  zz=z(st,mo); steps=sorted(set.intersection(*(set(r['curve']) for r in zz))); plt.plot(steps,[statistics.mean(r['curve'][s] for r in zz) for s in steps],color=colors[mo],label=mo)
 plt.xlabel('Optimizer step');plt.ylabel('Validation loss (lower is better)');plt.title(title);plt.legend();plt.grid(alpha=.2);plt.tight_layout();plt.savefig(fig/name,dpi=220);plt.close()
# delta curves
plt.figure(figsize=(8,5))
for st in ['concentrated','two_distributed','four_distributed']:
 a=sorted(z(st,'lora'),key=lambda r:r['seed']); b=sorted(z(st,'symmetric_quadratic'),key=lambda r:r['seed']); steps=sorted(set.intersection(*(set(r['curve']) for r in a+b))); plt.plot(steps,[statistics.mean(bb['curve'][s]-aa['curve'][s] for aa,bb in zip(a,b)) for s in steps],ls=ls[st],label=st)
plt.axhline(0,color='k',lw=.8);plt.xlabel('Optimizer step');plt.ylabel('ΔLoss = Symmetric − LoRA');plt.title('Architecture gap through 5,000 steps');plt.legend();plt.grid(alpha=.2);plt.tight_layout();plt.savefig(fig/'delta_loss_5000.png',dpi=220);plt.close()
# endpoint bars
plt.figure(figsize=(8,5)); labels=[]; x=[]; width=.35; idx=0
for st in ['concentrated','two_distributed','four_distributed']:
 for mo in ['lora','symmetric_quadratic']:
  a=next(x for x in agg if x['strategy']==st and x['model']==mo); x.append if False else None
# grouped manually
for i,st in enumerate(['concentrated','two_distributed','four_distributed']):
 a=next(x for x in agg if x['strategy']==st and x['model']=='lora'); b=next(x for x in agg if x['strategy']==st and x['model']=='symmetric_quadratic'); plt.bar(i-.18,a['val_mean'],.36,color=colors['lora'],label='LoRA' if i==0 else None);plt.bar(i+.18,b['val_mean'],.36,color=colors['symmetric_quadratic'],label='Symmetric' if i==0 else None)
plt.xticks(range(3),['concentrated','two-layer','four-layer']);plt.ylabel('Final validation loss');plt.title('5000-step endpoint validation loss');plt.legend();plt.tight_layout();plt.savefig(fig/'endpoint_validation_5000.png',dpi=220);plt.close()
# test bars
plt.figure(figsize=(8,5))
for i,st in enumerate(['concentrated','two_distributed','four_distributed']):
 a=next(x for x in agg if x['strategy']==st and x['model']=='lora'); b=next(x for x in agg if x['strategy']==st and x['model']=='symmetric_quadratic'); plt.bar(i-.18,a['test_mean'],.36,color=colors['lora'],label='LoRA' if i==0 else None);plt.bar(i+.18,b['test_mean'],.36,color=colors['symmetric_quadratic'],label='Symmetric' if i==0 else None)
plt.axhline(json.load(open(testdir/'baseline_test.json'))['test_loss'],color='k',ls='--',label='Frozen GPT-2 baseline');plt.xticks(range(3),['concentrated','two-layer','four-layer']);plt.ylabel('Test loss (lower is better)');plt.title('Held-out WikiText-2 test loss');plt.legend();plt.tight_layout();plt.savefig(fig/'test_loss_5000.png',dpi=220);plt.close()
# gap
plt.figure(figsize=(8,5))
for i,st in enumerate(['concentrated','two_distributed','four_distributed']):
 for j,mo in enumerate(['lora','symmetric_quadratic']):
  a=next(x for x in agg if x['strategy']==st and x['model']==mo); plt.bar(i-.18+j*.36,a['gap_mean'],.36,color=colors[mo],label=mo if i==0 else None)
plt.axhline(0,color='k',lw=.8);plt.xticks(range(3),['concentrated','two-layer','four-layer']);plt.ylabel('Test loss − validation loss');plt.title('Generalization gap');plt.legend();plt.tight_layout();plt.savefig(fig/'generalization_gap_5000.png',dpi=220);plt.close()
# output norm / grad plots from representative? metrics columns
plt.figure(figsize=(8,5))
for st in ['two_distributed','four_distributed']:
 for mo in ['lora','symmetric_quadratic']:
  r=z(st,mo)[0]; m=list(csv.DictReader(open(base/r['name']/'metrics.csv'))); key='adapter_adapter_output_norm';
  if key in m[0]: plt.plot([int(x['step']) for x in m],[float(x[key]) for x in m],label=f'{mo} {st}')
plt.xlabel('Optimizer step');plt.ylabel('Adapter output norm');plt.title('Adapter output norm (representative seed)');plt.legend(fontsize=8);plt.grid(alpha=.2);plt.tight_layout();plt.savefig(fig/'adapter_output_norm_5000.png',dpi=220);plt.close()
# report
lines=['# Long Convergence and Held-out Test Study','', '## Executive summary','', 'This study extends the fixed-budget 12,288-parameter comparison from 2,000 to 5,000 optimizer steps. It includes LoRA and Symmetric Quadratic in concentrated, two-layer, and four-layer geometries, with seeds 42, 123, and 456: 18/18 valid runs. Adapter-only checkpoints were saved at steps 1000, 2000, 3000, 4000, and 5000, and a reload smoke test reproduced logits with maximum absolute error 0.0.', '', 'The final checkpoint at step 5000 was chosen before looking at the public test set. Test evaluation used the same concatenation, GPT-2 tokenization, and 128-token block construction as validation, with no gradient updates. The full row-level results are in `long_results.csv`; aggregates are in `aggregate_long.csv`.','', '## Protocol and checkpointing','', 'The backbone is GPT-2 Small with frozen weights. Adapter parameters are FP32 while the backbone runs in FP16 autocast. Training uses WikiText-2, sequence length 128, batch size 1, accumulation 4, AdamW at 3e-4, and gradient clipping 1.0. Each run writes `config.json`, `metrics.csv`, `summary.json`, `run.log`, and adapter-only checkpoint files. Checkpoints contain adapter state and metadata; they do not duplicate GPT-2 weights.', '', '## Validation results at step 5000','', '| Geometry | Model | Val loss mean ± SD | PPL mean ± SD | AULC | Time s | Tokens/s | Test loss mean ± SD | Test PPL mean ± SD | Gap mean ± SD |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for a in agg: lines.append(f"| {a['strategy']} | {a['model']} | {a['val_mean']:.4f} ± {a['val_sd']:.4f} | {a['ppl_mean']:.2f} ± {a['ppl_sd']:.2f} | {a['aulc_mean']:.4f} | {a['time_mean']:.1f} | {a['tps_mean']:.0f} | {a['test_mean']:.4f} ± {a['test_sd']:.4f} | {a['test_ppl_mean']:.2f} ± {a['test_ppl_sd']:.2f} | {a['gap_mean']:.4f} ± {a['gap_sd']:.4f} |")
lines += ['', '## How to read the convergence figures','', 'Validation loss and test loss are lower-is-better. In ΔLoss figures, negative values favor Symmetric. AULC is the mean recorded validation loss over the stated interval; it summarizes the whole trajectory rather than a single endpoint.', '', '![Four layer](figures/validation_four_layer_5000.png)','', '**Observation.** The four-layer curves show how the 2,000-step ordering evolves through step 5,000. **Interpretation.** A changing gap indicates that the two parameterizations have different optimization trajectories; it does not by itself prove different representational limits.', '', '![Two layer](figures/validation_two_layer_5000.png)','', '**Observation.** The two-layer curves provide the controlled distributed comparison. **Interpretation.** Differences at 5,000 should be read together with seed SD and AULC.', '', '![Delta](figures/delta_loss_5000.png)','', '**Observation.** ΔLoss(t) directly shows whether the Symmetric advantage grows, shrinks, or changes sign. **Limitation.** Three seeds remain a small sample.', '', '## Held-out test methodology and results','', 'The frozen GPT-2 baseline was evaluated on 2,213 test blocks and obtained test loss 4.1279 (perplexity 62.05). Adapter test values are in the table above and in `test_eval/*.json`. The test checkpoint was always the final step-5000 checkpoint; validation was not used to choose among test results.', '', '![Test loss](figures/test_loss_5000.png)','', '**Observation.** Compare each adapter bar with the frozen baseline and with the corresponding validation result. **Interpretation.** Similar validation and test ordering supports transfer within WikiText-2; a changed ordering indicates a generalization difference, not automatically a training bug.', '', '![Generalization gap](figures/generalization_gap_5000.png)','', 'The generalization gap is `test loss − validation loss`; positive values indicate higher held-out loss.', '', '## Adapter and gradient diagnostics','', 'New metrics include adapter output norm, output-to-base ratio, transformed quadratic statistics, and gradient/weight norms with fully qualified parameter names. `adapter_output_norm_5000.png` is generated when the diagnostic column is available. These diagnostics describe utilization and optimization scale; they do not establish causality.', '', '## Interpretation','', 'The 5,000-step results answer whether the 2,000-step validation ordering persists under longer optimization and whether it reproduces on held-out text. Conclusions are limited to GPT-2 Small, WikiText-2, the attn.c_proj insertion points, the tested budgets, and this optimizer protocol. Three seeds provide replication but not a large-sample significance test.', '', '## Limitations and reproducibility','', 'The public test set was evaluated only after training and was not used for architecture, layer, rank, or checkpoint selection. The model is small, the dataset is narrow, and the run protocol uses one learning rate. PCIe/NVIDIA lockups occurred in earlier campaigns, but all 18 new 5,000-step runs completed after recovery. The exact configurations, checkpoints, metrics, test JSON files, and scripts are retained under `results_long_convergence_5000/`.']
(out/'LONG_CONVERGENCE_AND_TEST_STUDY.md').write_text('\n'.join(lines)+'\n')
print('long rows',len(rows),'aggregates',len(agg))
for a in agg: print(a['strategy'],a['model'],round(a['val_mean'],4),round(a['test_mean'],4),round(a['gap_mean'],4))
