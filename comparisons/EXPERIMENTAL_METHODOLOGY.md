# Experimental methodology comparison

| Method/work | Backbones and tasks in original work | Rank/placement/cost analyses | Status in this repository |
|---|---|---|---|
| LoRA | RoBERTa, DeBERTa, GPT-2, GPT-3; NLU and generation | ranks, selected projections, parameter/memory efficiency | reproduced as the controlled baseline |
| DoRA | LLaMA, LLaVA, VL-BART; language and vision-language tasks | LoRA-matched PEFT and training analysis | not tested |
| MoRA | LLM instruction, math, continual pretraining, memory, pretraining | higher-rank behavior at LoRA-like budgets | not tested |
| HiRA | Llama-2/3; commonsense, dialogue, math | rank, placement, ablation, speed/memory | methodology adapted, method not tested |
| PERA | LLaMA-family commonsense and RoBERTa/GLUE | rank, placement, square/cross ablation, interaction and cost | analysis ideas adapted to the activation-space formulation; method not tested |
| QuadraNet V2 | vision backbones/datasets | quadratic placement, factorization, sparse/atrous design, GPU-hour analysis | not tested; protocol incompatible with direct numeric comparison |
| This study | GPT-2 Small; WikiText-2 and AG News | three seeds, matched budgets, placement/depth, rank, scaling, convergence, derivatives, latency/memory | measured here |

## Reproduced or adapted methodology

- The matched-parameter LoRA baseline is reproduced directly.
- Rank and placement analyses follow common PEFT evaluation practice, with
  protocols frozen independently of the cited papers.
- PERA's emphasis on separating square/cross contributions motivated a related
  question, but this repository's ablations operate in activation space and do
  not reproduce PERA's factor-space ablation.
- The interaction analysis is original to the implemented map: analytical
  adapter Jacobians/Hessians and a fixed held-out sample. It is not a copied
  PERA metric or figure.
- Published results from other work are literature context only.
