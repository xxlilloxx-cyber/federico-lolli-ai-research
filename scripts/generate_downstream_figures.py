#!/usr/bin/env python3
import csv, pathlib, numpy as np, matplotlib.pyplot as plt
rows=list(csv.DictReader(open('docs/data/downstream_controlled_results.csv')))
out=pathlib.Path('docs/assets/publication');out.mkdir(parents=True,exist_ok=True)
for metric,label,stem in [('test_accuracy','Test accuracy','downstream_accuracy_comparison'),('test_macro_f1','Test macro-F1','downstream_macro_f1_comparison')]:
 fig,ax=plt.subplots(figsize=(6,4)); methods=['lora','symmetric_quadratic']; names=['LoRA','Symmetric Quadratic']; vals=[]
 for m in methods: vals.append(np.array([float(r[metric]) for r in rows if r['method']==m]))
 means=[x.mean() for x in vals]; sds=[x.std(ddof=1) for x in vals]; x=np.arange(2)
 ax.bar(x,means,yerr=sds,capsize=5,color=['#3b82f6','#c2410c'],alpha=.8,label='mean ± sample SD')
 for i,v in enumerate(vals): ax.scatter(np.full(3,i),v,color='black',zorder=3,label='seed result' if i==0 else None)
 ax.set_xticks(x,names);ax.set_ylabel(label);ax.set_ylim(0,1);ax.legend();ax.set_title('AG News controlled comparison (three seeds)');fig.tight_layout()
 for ext in ['svg','png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=220)
 plt.close(fig)
