# Qdrant Retrieval Test Report

- Checkpoint: `ml/checkpoints/jepa_best.pt` (best val loss 0.0447 at epoch 8)
- Embedding dim: **128**
- Points indexed: train=2989, val=640, test=640

## Latency (ms)

| stat | ms |
|---|---:|
| mean | 68.35 |
| median | 67.90 |
| p95 | 79.27 |
| p99 | 85.28 |

## Top-k intervention concordance

Majority-class baseline: **35.16%**

| k | any-hit | JEPA maj-vote | random maj-vote | lift | all-same | mean cos |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.50% | 32.50% | 36.41% | -3.91pp | 32.50% | +0.9929 |
| 5 | 86.25% | 34.06% | 32.34% | +1.72pp | 0.31% | +0.9903 |
| 10 | 98.12% | 36.56% | 35.16% | +1.41pp | 0.00% | +0.9888 |
| 25 | 100.00% | 36.56% | 32.81% | +3.75pp | 0.00% | +0.9864 |

## Per-intervention majority-vote accuracy (k=5)

- lisinopril: **30.10%**
- atorvastatin: **35.11%**
- metformin: **36.84%**

## Verdicts

- **GOOD** — [twin] Top-5 cosine high (patients cluster in embedding space): mean top-5 cosine = +0.990
- **GOOD** — [twin] Top-5 surfaces at least one same-intervention twin: any-match in top-5 = 86.2%
- **CONCERN** — [intervention] Top-5 majority-vote lifts over random retrieval: JEPA 34.1% vs random 32.3% (lift +1.7pp)
- **CONCERN** — [intervention] Top-10 majority-vote lifts over random retrieval: JEPA 36.6% vs random 35.2% (lift +1.4pp)
- **GOOD** — Query latency is interactive (filtered scan at 3k points): p95 = 79.3 ms (full_scan_threshold=10k, so filter => scan)