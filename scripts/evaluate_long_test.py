import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]; base=root/'results_long_convergence_5000'/'budget12288_5000'; out=root/'results_long_convergence_5000'/'test_eval'; out.mkdir(parents=True,exist_ok=True)
for d in sorted(base.iterdir()):
 if not d.is_dir() or not (d/'summary.json').exists(): continue
 name=d.name; dest=out/(name+'.json')
 if dest.exists(): print('SKIP',name,flush=True); continue
 cfg=d/'config.json'; ck=d/'checkpoints'/'adapter_final.pt';
 cmd=[str(root/'.venv/bin/python'),str(root/'scripts/evaluate_checkpoint.py'),'--config',str(cfg),'--checkpoint',str(ck),'--output',str(dest)]
 print('START',name,flush=True); r=subprocess.run(cmd,cwd=root,text=True); print('DONE',name,'rc',r.returncode,flush=True)
 if r.returncode!=0: raise SystemExit(1)
 h=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10); print('HEALTH',h.stdout.strip(),flush=True)
# frozen base once
b=out/'baseline_test.json'
if not b.exists():
 cfg={'model':'gpt2','sequence_length':128}; p=out/'baseline_config.json'; p.write_text(json.dumps(cfg)); r=subprocess.run([str(root/'.venv/bin/python'),str(root/'scripts/evaluate_checkpoint.py'),'--config',str(p),'--output',str(b),'--base'],cwd=root); print('BASE_DONE',r.returncode,flush=True)
