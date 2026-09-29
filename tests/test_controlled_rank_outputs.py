import csv,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def test_confirmation_grid_and_metrics():
 expected={(m,r,s) for m in ('LoRA','Symmetric Quadratic') for r in (1,2,4,8) for s in (42,123,456)}
 public=list(csv.DictReader(open(ROOT/'results/tables/rank-confirmation-5000/rank_5000_per_seed.csv')))
 assert {(x['method'],int(x['rank']),int(x['seed'])) for x in public}==expected
 assert all(math.isfinite(float(x[k])) for x in public for k in ('best_recorded_validation_loss','final_validation_loss','final_test_loss'))
 # Private checkpoints strengthen the local check when present, but are not
 # required for a clean public clone to run the test suite.
 private=ROOT/'results_rank_factorial_controlled'/'confirmation_5000'
 if private.exists():
  runs=[]
  for p in private.glob('*/summary.json'):
   s=json.loads(p.read_text()); d=p.parent
   assert (d/'config.json').exists() and (d/'metrics.csv').exists() and (d/'final_test.json').exists() and (d/'checkpoints'/'adapter_final.pt').exists()
   assert s['steps']==5000 and s['training_steps']==5000
   runs.append((('LoRA' if s['kind']=='lora' else 'Symmetric Quadratic'),int(s['rank']),int(s['seed'])))
  assert set(runs)==expected

def test_rank_aggregate_reconstructs_from_per_seed():
 p=list(csv.DictReader(open(ROOT/'docs/data/rank_5000_per_seed.csv')));a=list(csv.DictReader(open(ROOT/'docs/data/rank_5000_aggregate.csv')))
 for row in a:
  z=[float(x['final_validation_loss']) for x in p if x['method']==row['method'] and x['rank']==row['rank']]
  assert len(z)==3
  assert np.isclose(np.mean(z),float(row['final_validation_loss_mean']))
  assert np.isclose(np.std(z,ddof=1),float(row['final_validation_loss_sample_sd']))

def test_scaling_ablation_grid():
 rows=list(csv.DictReader(open(ROOT/'docs/data/rank_scaling_ablation_per_seed.csv')))
 policy=[x for x in rows if x['policy']=='constant_effective_scale']
 assert len(policy)==8
 assert {(x['method'],int(x['rank'])) for x in policy}=={(m,r) for m in ('LoRA','Symmetric Quadratic') for r in (1,2,4,8)}
 assert all(np.isclose(float(x['alpha_over_rank']),1.0) for x in policy)
