import csv, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapters import freeze_and_insert, parameter_counts, save_adapter_checkpoint


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed_all(cfg['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda')
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    model = AutoModelForCausalLM.from_pretrained(cfg['model'], torch_dtype=torch.float16).to(device)
    model.config.use_cache = False
    if cfg['kind'] != 'baseline':
        freeze_and_insert(model, cfg['kind'], cfg.get('rank', 4), cfg.get('alpha', 4.0),
                          tuple(cfg['layers']), tuple(cfg['projections']), True,
                          rank_l=cfg.get('rank_l'), rank_q=cfg.get('rank_q'),
                          alpha_l=cfg.get('alpha_l'), alpha_q=cfg.get('alpha_q'))
    else:
        for p in model.parameters(): p.requires_grad_(False)
    counts = parameter_counts(model)
    raw = load_dataset('wikitext', 'wikitext-2-raw-v1')
    seq = cfg['sequence_length']
    def blocks(split):
        ids = tok('\n'.join(x for x in raw[split]['text'] if x.strip()), add_special_tokens=False)['input_ids']
        return [torch.tensor(ids[i:i+seq], dtype=torch.long) for i in range(0, len(ids)-seq, seq)]
    train, valid = blocks('train'), blocks('validation')
    def loss_for(t):
        x = t.unsqueeze(0).to(device)
        with torch.autocast('cuda', dtype=torch.float16): return model(x, labels=x).loss
    out = Path(cfg.get('output_dir', f"results/{cfg['kind']}/r{cfg.get('rank', 0)}_seed{cfg['seed']}"))
    out.mkdir(parents=True, exist_ok=True); (out/'config.json').write_text(json.dumps(cfg, indent=2))
    checkpoint_steps = set(cfg.get('checkpoint_steps', [1000, 2000, 3000, 4000, 5000]))
    checkpoint_dir = out / 'checkpoints'
    if cfg['kind'] != 'baseline':
        checkpoint_dir.mkdir(exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad(): initial_val = float(torch.stack([loss_for(t).float() for t in valid[:8]]).mean())
    rows, nan_count, inf_count = [], 0, 0
    optimizer = None; scaler = None
    if cfg['kind'] != 'baseline':
        model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg['learning_rate'])
        scaler = torch.amp.GradScaler('cuda')
    start = time.monotonic(); total_tokens = 0
    eval_every = int(cfg.get('eval_every', 25))
    cached_val = initial_val
    for step in range(1, cfg['steps'] + 1):
        if optimizer:
            optimizer.zero_grad(set_to_none=True); grad_norm = 0.0
            for j in range(cfg['gradient_accumulation']):
                t = train[((step-1)*cfg['gradient_accumulation'] + j) % len(train)]
                loss = loss_for(t); total_tokens += len(t)
                if not torch.isfinite(loss):
                    nan_count += int(torch.isnan(loss)); inf_count += int(torch.isinf(loss)); raise FloatingPointError(f'nonfinite loss step {step}')
                scaler.scale(loss / cfg['gradient_accumulation']).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0))
            adapter_grad_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    key = name.replace('.', '_')
                    adapter_grad_stats[f'grad_norm_{key}'] = float(param.grad.detach().float().norm())
                    adapter_grad_stats[f'weight_norm_{key}'] = float(param.detach().float().norm())
            scaler.step(optimizer); scaler.update(); train_loss = float(loss.detach())
        else:
            train_loss = initial_val; grad_norm = 0.0
        if step == 1 or step % eval_every == 0 or step == cfg['steps']:
            model.eval()
            with torch.no_grad(): cached_val = float(torch.stack([loss_for(t).float() for t in valid[:8]]).mean())
        val_loss = cached_val
        model.train() if optimizer else None
        row = {'step': step, 'training_loss': train_loss, 'validation_loss': val_loss,
               'perplexity': math.exp(min(val_loss, 20)), 'gradient_norm': grad_norm,
               'learning_rate': optimizer.param_groups[0]['lr'] if optimizer else 0.0,
               'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'nan_count': nan_count, 'inf_count': inf_count}
        if optimizer:
            row.update(adapter_grad_stats)
        for module in model.modules():
            if hasattr(module, 'last_stats') and module.last_stats:
                for key, value in module.last_stats.items():
                    row[f'adapter_{key}'] = value
                break
        rows.append(row); print(f"{cfg['kind']} seed={cfg['seed']} step={step} train={train_loss:.4f} val={val_loss:.4f} peak_mb={row['peak_vram_bytes']/2**20:.1f}", flush=True)
        if optimizer and (step in checkpoint_steps or step == cfg['steps']):
            save_adapter_checkpoint(model, checkpoint_dir / f'adapter_step_{step:05d}.pt', config=cfg, optimizer=optimizer, step=step)
            (checkpoint_dir / f'adapter_step_{step:05d}.json').write_text(json.dumps({**cfg, 'checkpoint_step': step}, indent=2))
            print(f"CHECKPOINT step={step} path={checkpoint_dir / f'adapter_step_{step:05d}.pt'}", flush=True)
    elapsed = time.monotonic() - start
    with (out/'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    best = min(rows, key=lambda r: r['validation_loss']); last = rows[-1]
    summary = {**cfg, **counts, 'initial_validation_loss': initial_val,
               'validation_loss': best['validation_loss'], 'perplexity': best['perplexity'],
               'best_step': best['step'], 'peak_vram_bytes': max(r['peak_vram_bytes'] for r in rows),
               'training_seconds': elapsed, 'milliseconds_per_step': 1000*elapsed/max(1,cfg['steps']),
               'tokens_per_second': total_tokens/elapsed if elapsed else 0, 'nan_count': nan_count, 'inf_count': inf_count,
               'final_training_loss': last['training_loss'], 'parameter_difference_vs_lora_percent': cfg.get('parameter_difference_vs_lora_percent', 0.0)}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    if optimizer:
        # Ensure a clearly named final artifact even if steps was not in checkpoint_steps.
        final_path = checkpoint_dir / 'adapter_final.pt'
        if not final_path.exists() or cfg['steps'] not in checkpoint_steps:
            save_adapter_checkpoint(model, final_path, config=cfg, optimizer=optimizer, step=cfg['steps'])
        (checkpoint_dir / 'adapter_final.json').write_text(json.dumps({**cfg, 'checkpoint_step': cfg['steps']}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main(sys.argv[1])
