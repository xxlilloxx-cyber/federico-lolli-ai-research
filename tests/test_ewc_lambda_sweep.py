from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_ewc_lambda_sweep import DEFAULT_LAMBDAS,pareto,parse_lambdas,run_name


def test_predeclared_lambda_grid_and_override():
    assert DEFAULT_LAMBDAS==(0.0,1.0,10.0,30.0,50.0,100.0,300.0)
    assert parse_lambdas(None,None)==DEFAULT_LAMBDAS
    assert parse_lambdas(None,50)==(50.0,)
    assert parse_lambdas('0,10,300',None)==(0.0,10.0,300.0)


def test_run_names_are_isolated_and_stable():
    assert run_name('lora',0)=='lora_lambda_0_seed_42'
    assert run_name('combined',50)=='combined_lambda_50_seed_42'


def test_pareto_candidates_use_only_forgetting_and_acquisition():
    rows=[{'method':'lora','lambda':0.0,'mean_forgetting':4.0,'mean_new_fact_acquisition_nll':0.2},
          {'method':'lora','lambda':10.0,'mean_forgetting':2.0,'mean_new_fact_acquisition_nll':0.5},
          {'method':'lora','lambda':30.0,'mean_forgetting':3.0,'mean_new_fact_acquisition_nll':0.8},
          {'method':'symmetric','lambda':30.0,'mean_forgetting':1.0,'mean_new_fact_acquisition_nll':1.0}]
    found={(r['method'],r['lambda']) for r in pareto(rows)}
    assert found=={('lora',0.0),('lora',10.0),('symmetric',30.0)}
