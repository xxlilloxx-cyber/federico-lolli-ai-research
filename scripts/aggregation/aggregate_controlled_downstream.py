#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
root=Path('results_downstream_controlled/full'); out=Path('results/tables/downstream-ag-news');out.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(root.glob('*/summary.json')):
 s=json.loads(p.read_text()); rows.append({k:s.get(k) for k in ['method','seed','rank','placement','trainable_parameters','total_parameters','best_validation_loss','best_validation_accuracy','best_validation_macro_f1','final_validation_loss','final_validation_accuracy','final_validation_macro_f1','test_loss','test_accuracy','test_macro_f1','training_steps','training_time_seconds','peak_gpu_memory_mb']})
if not rows: raise SystemExit('no completed downstream runs')
with (out/'downstream_controlled_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
agg=[]
for method in sorted({r['method'] for r in rows}):
 rr=[r for r in rows if r['method']==method]
 if len(rr)!=3: continue
 d={'method':method,'n':3,'rank':rr[0]['rank'],'placement':rr[0]['placement'],'trainable_parameters':rr[0]['trainable_parameters']}
 for k in ['best_validation_loss','best_validation_accuracy','best_validation_macro_f1','final_validation_loss','final_validation_accuracy','final_validation_macro_f1','test_loss','test_accuracy','test_macro_f1']:
  x=np.array([float(r[k]) for r in rr]);d[k+'_mean']=x.mean();d[k+'_sample_sd']=x.std(ddof=1)
 agg.append(d)
if agg:
 with (out/'downstream_controlled_aggregate.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=agg[0]);w.writeheader();w.writerows(agg)
lor={int(r['seed']):r for r in rows if r['method']=='lora'}; sym={int(r['seed']):r for r in rows if r['method']=='symmetric_quadratic'}
if set(lor)==set(sym)=={42,123,456}:
 dif=[]
 for seed in sorted(lor):
  dif.append({'seed':seed,'symmetric_minus_lora_test_accuracy':float(sym[seed]['test_accuracy'])-float(lor[seed]['test_accuracy']),'symmetric_minus_lora_test_macro_f1':float(sym[seed]['test_macro_f1'])-float(lor[seed]['test_macro_f1']),'symmetric_minus_lora_test_loss':float(sym[seed]['test_loss'])-float(lor[seed]['test_loss'])})
 with (out/'downstream_controlled_paired_differences.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=dif[0]);w.writeheader();w.writerows(dif)
print(f'per-seed={len(rows)} aggregates={len(agg)}')
