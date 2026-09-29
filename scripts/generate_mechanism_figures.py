#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
out=Path('docs/assets/publication');out.mkdir(parents=True,exist_ok=True)
j=pd.read_csv('docs/data/jacobian_analysis.csv');h=pd.read_csv('docs/data/interaction_analysis.csv')
plt.style.use('seaborn-v0_8-whitegrid')
for metric,label,name in [('effective_rank','Local Jacobian effective rank','jacobian_effective_rank'),('spectral_norm','Local Jacobian spectral norm','jacobian_norm')]:
 f,a=plt.subplots(figsize=(7,4.5))
 g=j.groupby(['method','rank'])[metric].agg(['mean','std']).reset_index()
 for method,color in [('LoRA','#2563eb'),('Symmetric Quadratic','#c2410c')]:
  z=g[g.method==method];a.errorbar(z['rank'],z['mean'],yerr=z['std'],marker='o',capsize=4,color=color,label=method+' mean ± sample SD')
 a.set_xscale('log',base=2);a.set_xticks([1,2,4,8],['1','2','4','8']);a.set_xlabel('Rank');a.set_ylabel(label);a.legend();a.set_title('Fixed held-out WikiText-2 activation sample');f.tight_layout()
 for e in ('svg','png','pdf'):f.savefig(out/(name+'.'+e),dpi=240)
 plt.close(f)
f,a=plt.subplots(figsize=(7,4.5));g=h.groupby(['method','rank'])['hessian_frobenius_norm'].agg(['mean','std']).reset_index()
for method,color in [('LoRA','#2563eb'),('Symmetric Quadratic','#c2410c')]:
 z=g[g.method==method];a.errorbar(z['rank'],z['mean'],yerr=z['std'],marker='o',capsize=4,color=color,label=method)
a.set_xscale('log',base=2);a.set_xticks([1,2,4,8],['1','2','4','8']);a.set_xlabel('Rank');a.set_ylabel('Adapter-only Hessian Frobenius norm');a.legend();a.set_title('Fixed residual-direction scalar; mean ± seed SD');f.tight_layout()
for e in ('svg','png','pdf'):f.savefig(out/('adapter_interaction_hessian.'+e),dpi=240)
