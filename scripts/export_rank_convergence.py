#!/usr/bin/env python3
import csv
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
root=Path('results_rank_factorial_controlled/confirmation_5000');rows=[]
for p in root.glob('*/summary.json'):
 import json
 s=json.loads(p.read_text()); method='LoRA' if s['kind']=='lora' else 'Symmetric Quadratic'
 for x in csv.DictReader(open(p.parent/'metrics.csv')):
  step=int(x['step'])
  if step==1 or step%25==0:rows.append({'method':method,'rank':int(s['rank']),'seed':int(s['seed']),'step':step,'validation_loss':float(x['validation_loss'])})
d=pd.DataFrame(rows);d.to_csv('docs/data/rank_5000_convergence.csv',index=False)
out=Path('docs/assets/publication');plt.style.use('seaborn-v0_8-whitegrid')
for rank in (1,2,4,8):
 f,a=plt.subplots(figsize=(7,4.5));z=d[d['rank']==rank]
 for method,color in [('LoRA','#2563eb'),('Symmetric Quadratic','#c2410c')]:
  q=z[z.method==method].groupby('step').validation_loss.agg(['mean','std']);a.plot(q.index,q['mean'],color=color,label=method);a.fill_between(q.index,q['mean']-q['std'],q['mean']+q['std'],color=color,alpha=.15,label=method+' sample SD')
 a.set_xlabel('Optimizer step');a.set_ylabel('Fresh validation loss');a.set_title(f'5,000-step convergence, rank {rank}');a.legend(fontsize=8);f.tight_layout()
 for e in ('svg','png','pdf'):f.savefig(out/(f'convergence_rank{rank}.'+e),dpi=240)
 plt.close(f)
# paired mean crossover report
for rank in (1,2,4,8):
 p=d[d['rank']==rank].pivot_table(index=['seed','step'],columns='method',values='validation_loss').reset_index();p['delta']=p['Symmetric Quadratic']-p['LoRA'];m=p.groupby('step').delta.mean();cross=[]
 for a,b in zip(m.index[:-1],m.index[1:]):
  if m[a]*m[b]<0:cross.append((int(a),int(b)))
 print(rank,cross,'delta1000',m.get(1000),'delta5000',m.get(5000))
