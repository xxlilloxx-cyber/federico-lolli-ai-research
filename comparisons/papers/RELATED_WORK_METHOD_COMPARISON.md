# Related Work and Method Comparison

This table is literature context. Published results use different backbones,
datasets, parameter budgets, and evaluation rules and are not numerically
comparable with this repository's GPT-2 experiments.

| Method | Mathematical / architectural mechanism | Evaluation reported by original work | Relation to this project | Reproduced here? |
|---|---|---|---|---|
| LoRA | Frozen weight plus low-rank weight update `ΔW=AB`; mergeable at inference | RoBERTa, DeBERTa, GPT-2 and GPT-3 across NLU and generation; task metrics and parameter/memory analyses | Direct controlled baseline, implemented as `Δy=(alpha/r)(xA)B` | Yes |
| Weight-Decomposed DoRA | Decomposes pretrained weights into magnitude and direction; LoRA updates direction | LLaMA, LLaVA and VL-BART; commonsense reasoning, visual instruction tuning, image/video-text tasks | Weight-space reparameterization; no activation-quadratic branch | No |
| MoRA | Trainable square matrix with fixed non-parametric compression/expansion operators to obtain higher-rank updates at LoRA-like budgets | Instruction tuning, mathematical reasoning, continual pretraining, memory and pretraining | High-rank weight-space update rather than activation-dependent quadratic mapping | No |
| HiRA | Hadamard product formulation intended to retain high-rank update capacity at small trainable budgets | Llama-2-7B and Llama-3-8B; commonsense, ConvAI2 and mathematical reasoning; rank, placement and cost ablations | Uses multiplicative weight-update structure; it does not implement `(xU)^2P` | No |
| LoRAN | Applies a nonlinear map to the accumulated low-rank weight update, commonly written `ΔW=f(BA)` | SAMSum and 20 Newsgroups; summarization/classification metrics, including low-rank comparisons | Nonlinearity is in weight-update space, unlike the activation-space correction here | No |
| PERA | Polynomially expands low-rank factors before composition, including original, square and cross components | LLaMA-family commonsense reasoning and RoBERTa/GLUE in the paper; rank robustness, placement, square/cross ablations, feature-interaction and efficiency analyses | Closest polynomial motivation, but polynomial terms are constructed in factor/parameter space rather than from current input activations | No |
| Symmetric Quadratic (this study) | `Δy=(alpha/r)((xU)⊙(xU))P`; local Jacobian is input-dependent and adapter Hessian can be non-zero | GPT-2/WikiText-2 loss and perplexity; GPT-2/AG News accuracy and macro-F1; controlled rank, placement, scaling, mechanism and cost analyses | Independently evaluated activation-space quadratic adapter | Yes |

## Methodological mapping

| Analysis methodology | Source precedent | Status in this project | Difference |
|---|---|---|---|
| Matched-parameter LoRA baseline | LoRA and later PEFT work | Reproduced | GPT-2 Small and one/few insertion points |
| Rank robustness | LoRA, HiRA, PERA | Adapted | ranks 1/2/4/8, three seeds, WikiText-2, exact final/test separation |
| Placement study | HiRA and PERA | Adapted | GPT-2 depth/module screens and fixed-budget distribution |
| Square/cross ablation | PERA | Partially adapted | distinct activation-space elementwise, feature-interaction and symmetric families; not PERA's factor expansion |
| Feature-interaction analysis | PERA | Adapted conceptually | original analytical Hessian and local-Jacobian analysis appropriate to `(xU)^2P` |
| Memory/speed analysis | LoRA-family papers | Reproduced in a focused rank-4 benchmark | one RTX A500 environment, 30 measured iterations |
| Large-model commonsense/GLUE comparison | PERA/HiRA/DoRA | Not reproduced | hardware and model scope differ; external values are not mixed with our results |

## Verified references

1. Edward J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9
2. Shih-Yang Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. https://arxiv.org/abs/2402.09353
3. Ting Jiang et al. *MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning.* arXiv:2405.12130 (2024). https://arxiv.org/abs/2405.12130
4. Qiushi Huang et al. *HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models.* ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html
5. Yinqiao Li, Linqi Song, and Hanxu Hou. *LoRAN: Improved Low-Rank Adaptation by a Non-Linear Transformation.* Findings of EMNLP 2024, pp. 3134–3143. https://aclanthology.org/2024.findings-emnlp.177/
6. Wenhao Zhang, Lin Mu, Li Ni, Peiquan Jin, and Yiwen Zhang. *Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions.* Findings of ACL 2026, pp. 13287–13303, DOI 10.18653/v1/2026.findings-acl.650. https://aclanthology.org/2026.findings-acl.650/

The PERA venue/status is supported by the ACL record and the authors' public
repository. No novelty claim is made merely from a different implementation.
