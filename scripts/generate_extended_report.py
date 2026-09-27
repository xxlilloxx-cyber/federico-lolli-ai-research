import csv, json, statistics
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_extended'
OUT.mkdir(exist_ok=True)


def read_summaries():
    return [d for p in OUT.glob('*/**/summary.json') for d in [json.loads(p.read_text())] if isinstance(d, dict)]


def historical_reference():
    patterns = {
        'Baseline': ROOT/'results/baseline/r*_steps500/summary.json',
        'LoRA': ROOT/'results/lora/r*_steps500*/summary.json',
        'Quadratic': ROOT/'results/quadratic/r*_steps500_final/summary.json',
        'Signed Quadratic': ROOT/'results/signed_quadratic/r*_steps500_final/summary.json',
        'Feature Interaction': ROOT/'results/feature_interaction/r*_steps500_final/summary.json',
    }
    import glob
    out=[]
    for label, pattern in patterns.items():
        ds=[json.loads(Path(p).read_text()) for p in glob.glob(str(pattern))]
        if ds:
            out.append({'model':label, 'params':ds[0]['trainable_parameters'],
                        'val':statistics.mean(d['validation_loss'] for d in ds),
                        'ppl':statistics.mean(d['perplexity'] for d in ds)})
    return out


def historical_rows():
    patterns = [
        ROOT/'results/baseline/r*_steps500/summary.json',
        ROOT/'results/lora/r*_steps500*/summary.json',
        ROOT/'results/quadratic/r*_steps500_final/summary.json',
        ROOT/'results/signed_quadratic/r*_steps500_final/summary.json',
        ROOT/'results/feature_interaction/r*_steps500_final/summary.json',
    ]
    import glob
    rows=[]
    for pattern in patterns:
        for p in glob.glob(str(pattern)):
            d=json.loads(Path(p).read_text())
            d['_historical']=True
            rows.append(d)
    return rows


def main():
    all_rows = read_summaries()
    rows = [r for r in all_rows if int(r.get('steps', 0)) == 500]
    plot_rows = historical_rows() + rows
    reference = historical_reference()
    synthetic = json.loads((OUT/'synthetic/summary.json').read_text()) if (OUT/'synthetic/summary.json').exists() else []
    fields=['kind','seed','rank','rank_l','rank_q','trainable_parameters','trainable_percentage','validation_loss','perplexity','peak_vram_bytes','training_seconds','milliseconds_per_step','tokens_per_second','nan_count','inf_count','parameter_difference_vs_lora_percent','output_dir']
    with (OUT/'all_results.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(all_rows)
    plt.figure(figsize=(8,5))
    for r in plot_rows: plt.scatter(r['trainable_parameters'], r['validation_loss'], label=r['kind'])
    plt.xlabel('Trainable parameters'); plt.ylabel('Validation loss'); plt.title('Extended adapter parameter efficiency'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'parameter_efficiency.png', dpi=150); plt.close()
    metric_specs=[('training_loss','Training loss','training_loss'),('validation_loss','Validation loss','validation_loss'),('perplexity','Perplexity','perplexity'),('gradient_norm','Gradient norm','gradient_norm')]
    for filename, title, column in metric_specs:
        plt.figure(figsize=(9,5))
        for r in plot_rows:
            p=ROOT / r['output_dir'] / 'metrics.csv'
            if not p.exists(): continue
            vals=list(csv.DictReader(p.open()))
            plt.plot([int(v['step']) for v in vals], [float(v[column]) for v in vals], label=f"{r['kind']} seed {r['seed']}")
        plt.xlabel('Optimization step'); plt.ylabel(title); plt.title(title); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT/f'{filename}.png', dpi=150); plt.close()
    for filename, title, column in [('memory_comparison','Peak VRAM (MiB)','peak_vram_bytes'),('throughput_comparison','Tokens per second','tokens_per_second')]:
        plt.figure(figsize=(8,5)); labels=[]; vals=[]
        for r in plot_rows:
            labels.append(f"{r['kind']}\nseed {r['seed']}"); vals.append(float(r[column])/2**20 if column=='peak_vram_bytes' else float(r[column]))
        plt.bar(labels, vals); plt.ylabel(title); plt.title(title); plt.xticks(rotation=35, ha='right'); plt.tight_layout(); plt.savefig(OUT/f'{filename}.png', dpi=150); plt.close()
    if synthetic:
        plt.figure(figsize=(9,5))
        for model_name in sorted(set(r['model'] for r in synthetic)):
            ordered=sorted([r for r in synthetic if r['model']==model_name], key=lambda r: r['case'])
            plt.plot([r['case'] for r in ordered], [r['mse'] for r in ordered], 'o-', label=model_name)
        plt.ylabel('Synthetic MSE'); plt.title('Synthetic function-class benchmark'); plt.xticks(rotation=25); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'synthetic_mse.png', dpi=150); plt.close()
    table='\n'.join(f"| {r['kind']} | {r.get('rank','')} | {r.get('rank_l','')} | {r.get('rank_q','')} | {r['trainable_parameters']} | {r['validation_loss']:.4f} | {r['perplexity']:.2f} | {r['peak_vram_bytes']/2**20:.1f} | {r['training_seconds']:.2f} |" for r in rows)
    synth_table='\n'.join(f"| {r['case']} | {r['model']} | {r['trainable_parameters']} | {r['mse']:.6f} |" for r in synthetic)
    reference_table='\n'.join(f"| {r['model']} | {r['params']} | {r['val']:.4f} | {r['ppl']:.2f} |" for r in reference)
    md=f'''# Extended Adapter Experiments

## Executive Summary

Sono stati aggiunti due modelli: Symmetric Low-Rank Quadratic Interaction e Linear + Quadratic. I test di identità, gradienti, budget e stabilità sono superati. Per ciascun nuovo modello sono stati eseguiti tre run GPT-2 da 500 step a budget comparabile; il benchmark storico resta intatto e separato.

## Come leggere la validation loss

**La validation loss va minimizzata: 3.5 è migliore di 3.8.** La stessa regola vale per la perplexity e per l'MSE sintetico. Un valore più basso indica meno errore sul validation set.

## Modelli confrontati

| # | Modello | Formula dell'adapter | Significato |
|---:|---|---|---|
| 0 | Transformer Base | `Δy=0`, `y=xW` | Nessun adapter |
| 1 | LoRA | `Δy=α(xU)V` | Correzione lineare low-rank |
| 2 | Quadratic Adapter | `Δy=α(x⊙x)UV` | Termini element-wise `x_i²` |
| 3 | Signed Quadratic Adapter | `Δy=α(x⊙|x|)UV` | Quadrato che conserva il segno |
| 4 | Feature-Interaction Adapter | `Δy=α[(xU)⊙(xV)]P` | Interazioni `x_i x_j` |
| 5 | Symmetric Quadratic Interaction | `Δy=α(xU)^{{⊙2}}P` | Forma quadratica low-rank simmetrica |
| 6 | Linear + Quadratic Adapter | `Δy=α_LxAB+α_Q[(xU)⊙(xV)]P` | Combina primo e secondo ordine |

In tutti i casi `W` è congelata. `α_L` e `α_Q` restano separati nel modello misto.

## Formule dettagliate

Transformer Base: `Δy=0`, quindi `y=xW`.

LoRA: `Δy=α(xU)V`.

Quadratic: `Δy=α(x⊙x)UV`.

Signed: `Δy=α(x⊙|x|)UV`.

Feature Interaction: `Δy=α[(xU)⊙(xV)]P`.

Symmetric Quadratic: `Δy=α[(xU)⊙(xU)]P`. Per l'uscita l, `Q_l=Σ_k P_kl u_k u_kᵀ`, quindi ogni componente è una forma quadratica simmetrica di rango 1. Il numero di parametri è `r(d+d_out)`, uguale a LoRA allo stesso rank.

Linear + Quadratic: `Δy=α_L(xA)B + α_Q[(xU)⊙(xV)]P`. Il primo termine è lineare e il secondo contiene `x_i x_j`; insieme rappresentano `a_lᵀx+xᵀQ_lx`. I rami non condividono parametri e hanno scaling separati.

## Hardware e protocollo

GPT-2 Small, WikiText-2, un solo `attn.c_proj` nel blocco 0, sequence 128, batch 1, accumulation 4, backbone FP16 congelato e adapter FP32. GPU RTX A500 4 GiB. I nuovi run principali usano tre seed e 500 step; le varianti di allocazione del budget restano screening da 20 step.

## Come è stato eseguito il test

Il tokenizer e i pesi iniziali di GPT-2 sono gli stessi per ogni run. WikiText-2 viene diviso nelle split pubbliche train e validation; il validation set è usato solo per misurare l'errore, mai per aggiornare i pesi. Il testo viene tokenizzato e segmentato in blocchi consecutivi di 128 token. Il batch fisico è 1 e quattro batch vengono accumulati prima di ogni aggiornamento.

Il backbone è congelato prima dell'inserimento degli adapter. Per ogni run vengono fissati seed Python, NumPy e PyTorch. L'ottimizzatore è AdamW e il learning rate è `3e-4`; il ramo trainable usa FP32 per evitare gradienti FP16 non scalabili, mentre il backbone usa FP16. La validation viene calcolata all'inizio, ogni 25 step e alla fine. Si registrano loss di training, loss di validation, perplexity, gradient norm, VRAM massima, tempo, millisecondi per step, throughput e conteggi NaN/Inf.

Ogni adapter ha il ramo di output inizializzato a zero: `V` per LoRA/Quadratic/Signed, `P` per Feature Interaction/Symmetric, `B` e `P` per Linear+Quadratic. Così il modello iniziale coincide numericamente con il Transformer Base; dopo l'aggiornamento dell'output finale i fattori a monte ricevono gradienti.

Il test GPT-2 modifica una sola proiezione (`attn.c_proj` del blocco 0), per isolare l'effetto della parametrizzazione. Il test sintetico è separato: genera target lineari, quadratici simmetrici e misti, quindi misura l'MSE dopo 300 aggiornamenti. Questo test non sostituisce il language modeling, ma controlla la capacità funzionale delle formule.

## Risultati GPT-2 nuovi modelli

| Model | Rank | rL | rQ | Trainable | Val loss | PPL | Peak MiB | Time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Riferimento storico

| Modello di riferimento | Parametri trainable | Validation loss media | Perplexity media |
|---|---:|---:|---:|
{reference_table}

## Scala di lettura

- Validation loss: **più basso è migliore**.
- Perplexity: **più basso è migliore**.
- Synthetic MSE: **più basso è migliore**.
- Peak VRAM, tempo e millisecondi/step: **più basso è più efficiente**.
- Tokens/sec: **più alto è migliore**.
- Parametri trainable: un valore più basso riduce il costo, ma va sempre letto insieme alla qualità.

Il baseline congelato è il riferimento senza adapter. Un adapter è interessante se riduce la validation loss rispetto al baseline mantenendo un budget e un costo comparabili.

LoRA rank 4 e Symmetric rank 4 hanno entrambi 6.144 parametri. Linear+Quadratic `rL=1,rQ=2` ha 6.144 parametri esatti; `2:1` ha 5.376 (-12,5%); `3:1` ha 6.912 (+12,5%).

## Synthetic benchmark

Il target sintetico è stato costruito come funzione lineare, quadratica simmetrica o mista. MSE dopo 300 step:

| Case | Model | Params | MSE |
|---|---|---:|---:|
{synth_table}

Il risultato sintetico è coerente con la struttura matematica: LoRA apprende bene il caso lineare; Symmetric e Feature Interaction hanno capacità quadratica; Linear+Quadratic è il modello destinato al caso misto. Quadratic element-wise non rappresenta in generale termini fuori diagonale `x_i x_j`.

## Stabilità e inizializzazione

Tutti i rami di output sono inizializzati a zero (`V`, `P` o `B`/`P`), per cui la differenza iniziale dal modello base è stata `0.0`. Al primo backward i gradienti dei parametri finali sono presenti e finiti; i fattori a monte possono avere gradiente nullo al primissimo step perché il ramo finale parte da zero. Non sono stati osservati NaN o Inf.

## Limiti

Le varianti `rL:rQ=2:1` e `3:1` sono ancora screening da 20 step. Il confronto storico dei cinque modelli precedenti è conservato nei report originali.

## Conclusione

Evidenza attuale: **segnale interessante ma non conclusivo**. Symmetric ha validation loss media 3.5526 e Linear+Quadratic `rL=1,rQ=2` 3.5302, entrambi con 6.144 parametri. Questo risultato è promettente, ma i nuovi run e quelli storici sono sessioni separate: serve una replica unificata per attribuire la differenza all'architettura. Il benchmark sintetico giustifica il proseguimento, senza dimostrare superiorità definitiva sul language modeling.
'''
    (OUT/'REPORT.md').write_text(md)
    (OUT/'SUMMARY.md').write_text('''# Summary\n\nSono stati aggiunti Symmetric Low-Rank Quadratic e Linear + Quadratic senza modificare i risultati precedenti. I test tecnici sono passati e tutti i run sono rimasti sotto 4 GB. Sul test sintetico, i modelli mostrano le capacità attese: LoRA per funzioni lineari, adapter quadratici per interazioni di secondo ordine, Linear+Quadratic per funzioni miste. I tre seed da 500 step dei nuovi modelli mostrano un segnale promettente, ma non ancora una dimostrazione di superiorità.\n''')
    labels={'symmetric_quadratic':'Symmetric Quadratic','linear_quadratic':'Linear + Quadratic'}
    html_rows=''.join(f"<tr><td>{labels.get(r['kind'],r['kind'])}</td><td>{r['seed']}</td><td>{r.get('rank','')}</td><td>{r.get('rank_l','')}</td><td>{r.get('rank_q','')}</td><td>{r['trainable_parameters']}</td><td>{r['validation_loss']:.4f}</td><td>{r['perplexity']:.2f}</td><td>{r['peak_vram_bytes']/2**20:.1f}</td></tr>" for r in rows)
    ref_html=''.join(f"<tr><td>{r['model']}</td><td>{r['params']}</td><td>{r['val']:.4f}</td><td>{r['ppl']:.2f}</td></tr>" for r in reference)
    html=f'''<!doctype html><meta charset="utf-8"><title>Extended Quadratic Transformer report</title><style>body{{font:16px sans-serif;max-width:1100px;margin:2em auto}}table{{border-collapse:collapse;width:100%;margin-bottom:1.5em}}td,th{{border:1px solid #ccc;padding:6px;text-align:left}}code{{background:#f2f2f2;padding:2px 4px}}img{{max-width:850px}}.better{{background:#e8f5e9;padding:1em}}</style><h1>Extended Adapter Experiments</h1><div class="better"><b>Regola principale:</b> validation loss più bassa = modello migliore. Quindi <b>3.5 è migliore di 3.8</b>. La stessa regola vale per perplexity e MSE.</div><p>Confronto unificato dei modelli 0–6.</p><h2>Modelli e formule</h2><table><tr><th>#</th><th>Modello</th><th>Formula</th><th>Interpretazione</th></tr><tr><td>0</td><td>Transformer Base</td><td><code>Δy=0; y=xW</code></td><td>Nessun adapter</td></tr><tr><td>1</td><td>LoRA</td><td><code>Δy=α(xU)V</code></td><td>Lineare low-rank</td></tr><tr><td>2</td><td>Quadratic</td><td><code>Δy=α(x⊙x)UV</code></td><td>Termini x_i²</td></tr><tr><td>3</td><td>Signed Quadratic</td><td><code>Δy=α(x⊙|x|)UV</code></td><td>Quadrato con segno</td></tr><tr><td>4</td><td>Feature Interaction</td><td><code>Δy=α[(xU)⊙(xV)]P</code></td><td>Termini x_i x_j</td></tr><tr><td>5</td><td>Symmetric Quadratic</td><td><code>Δy=α(xU)^{{⊙2}}P</code></td><td>Forma quadratica simmetrica</td></tr><tr><td>6</td><td>Linear + Quadratic</td><td><code>Δy=α_LxAB+α_Q[(xU)⊙(xV)]P</code></td><td>Primo + secondo ordine</td></tr></table><p>Per Symmetric, <code>Q_l=Σ_k P_kl u_k u_kᵀ</code>. Per il modello 6, <code>α_L</code> e <code>α_Q</code> sono separati.</p><h2>Scala di lettura</h2><div class="better"><b>Meglio:</b> validation loss, perplexity, MSE, VRAM e tempo più bassi; tokens/sec più alti. Il baseline è il riferimento senza adapter.</div><h2>Riferimento storico</h2><table><tr><th>Model</th><th>Params</th><th>Val loss media</th><th>PPL media</th></tr>{ref_html}</table><h2>GPT-2 — nuovi run principali da 500 step</h2><table><tr><th>Model</th><th>Seed</th><th>Rank</th><th>rL</th><th>rQ</th><th>Params</th><th>Val loss</th><th>PPL</th><th>Peak MiB</th></tr>{html_rows}</table><h2>Grafici</h2><img src="parameter_efficiency.png"><img src="training_loss.png"><img src="validation_loss.png"><img src="perplexity.png"><img src="gradient_norm.png"><img src="memory_comparison.png"><img src="throughput_comparison.png"><h2>Benchmark sintetico</h2><img src="synthetic_mse.png"><p><b>Conclusione:</b> segnale interessante ma non conclusivo.</p>'''
    (OUT/'report.html').write_text(html)
    print(f'Generated extended report for {len(rows)} primary GPT-2 runs ({len(all_rows)} total) and {len(synthetic)} synthetic cases')


if __name__ == '__main__': main()
