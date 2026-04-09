# Cohort Compass — Executive Technical Summary

**Audience:** Engineers joining the project who need a fast, accurate mental model of the ML core and the product.

---

## 1. What the project is

**Cohort Compass** is a **causal healthcare analytics platform** that combines:

- A **clinical time-series representation model** (JEPA) that maps a patient’s history into a **latent “origin” embedding** and can **predict how that embedding would move under different interventions** (counterfactual trajectories in embedding space — not raw lab reconstruction).
- **Vector search** over historical patients (Qdrant) to find **similar “twins”** and ground predictions in real outcomes.
- **LLM/RAG** layers that turn structured results into clinician-facing narrative.
- A **Next.js** UI (galaxy / neighborhood visualization) and a **FastAPI** backend with **async jobs** (Celery) for end-to-end analysis.

**Primary data source for training cohorts:** All of Us (AoU) OMOP-style extracts in the Researcher Workbench (BigQuery CDR), with an index event (e.g., first drug exposure) and ±365 day windows for pre/post labs and vitals.

---

## 2. The JEPA model (`CausalJEPA`)

### 2.1 Idea in one sentence

The model learns **joint embeddings** so that **context (pre-intervention series) + intervention** predicts a **target representation (post-intervention series)** in latent space, using **MSE in latent space** — **no decoder** and **no reconstruction loss** on raw clinical values.

### 2.2 Components (implementation: `backend/ml/jepa_model.py`)

| Piece | Role |
|--------|------|
| **`TimeSeriesEncoder` (context)** | Transformer over multivariate time-series with **value + observation mask** per feature and **continuous time** (seconds-based sinusoidal PE). Produces a **pooled vector** \(s_x\) per patient. **Trained with gradients.** |
| **`TimeSeriesEncoder` (target)** | **Same architecture**, weights updated **only via EMA** from the context encoder (`requires_grad=False`). Produces \(s_y\) as a **stable target anchor**. |
| **`action_embedding`** | `nn.Embedding(K, z_dim)` — categorical intervention ID → dense vector. |
| **`Predictor`** | MLP: **concat(\(s_x\), embed(\(z\)))** → **predicted** \(\hat{s}_y\) with same dimension as \(s_y\). |

### 2.3 Training mechanics (when `backend/ml/train.py` is fully wired)

1. Forward: \(\hat{s}_y, s_y = \text{model}(\text{context\_x}, \text{target\_y}, \text{intervention\_z})\).
2. **Loss:** `MSE(predicted_s_y, s_y)` — \(s_y\) is detached; **no gradient through the target encoder** from the loss.
3. **Optimizer** updates **context encoder**, **action embedding**, and **predictor** only.
4. After each step: **`update_target_encoder(momentum)`** — EMA sync: \(\theta_{\text{target}} \leftarrow m\theta_{\text{target}} + (1-m)\theta_{\text{context}}\).

This follows the **I-JEPA / JEPA** pattern: predict in **representation space**, avoid collapse via predictor bottleneck + EMA target.

### 2.4 Training tensors (batch contract)

| Symbol | Tensor | Shape (conceptual) | Meaning |
|--------|--------|--------------------|---------|
| **\(x\)** | `context_x` | `(B, T_pre, F)` | Pre-index clinical values (labs/vitals aligned to a fixed feature vocabulary). |
| | `context_mask` | `(B, T_pre, F)` | 1 = observed, 0 = missing. |
| | `context_timestamps` | `(B, T_pre)` | Seconds from window start (irregular sampling). |
| **\(y\)** | `target_y` | `(B, T_post, F)` | Post-index window (same feature dim **F**). |
| | `target_mask`, `target_timestamps` | same pattern | |
| **\(z\)** | `intervention_z` | `(B,)` **LongTensor** | Integer ID in `0 … K-1` (e.g., drug / protocol class). |

**Outputs of the forward pass:** `predicted_s_y`, `s_y` each **`(B, d_model)`** (default `d_model=256`).

### 2.5 Inference helpers (deployment)

- **`encode(values, mask, timestamps, …)`** → **origin embedding** `(B, d_model)` for indexing / UI.
- **`predict_counterfactual(origin_embedding, intervention_z)`** → predicted latent trajectory for a **given** intervention ID (used with the causal engine + Qdrant).

**Note:** `ClinicalJEPA` is an **alias** for `CausalJEPA` for backward compatibility with existing imports.

---

## 3. End-to-end product I/O (what users and APIs see)

### 3.1 Typical API input (high level)

Defined in `backend/schemas/inference.py` and related patient schemas:

| Input | Description |
|--------|-------------|
| **`clinical_time_series`** | Structured series: observations with timestamps, concept IDs (OMOP/FHIR-like), values, feature ordering — see `ClinicalTimeSeries` in `backend/schemas/patient.py`. |
| **`demographics`** | Optional: age, sex, etc., for ranking and UI. |
| **`requested_interventions`** | Optional list of intervention **codes** to simulate; default = full registry in `services/counterfactual.py`. |

### 3.2 Typical API output (high level)

| Output | Description |
|--------|-------------|
| **Per-intervention simulation** | Latent **predicted embedding**, mapped UI coordinates, confidence and risk deltas — see `SimulatedTrajectory` in `schemas/inference.py`. |
| **Twin matches** | Similar historical patients from the vector DB (`TwinMatch`). |
| **Ranked / narrative layer** | RAG-generated summaries and neighborhood payload for the 3D/galaxy UI. |

So: **inputs** are **clinical time series (+ optional demographics + intervention subset)**; **outputs** are **counterfactual latent trajectories, similarity-based evidence, and text** — not raw reconstructed labs.

---

## 4. Repo map (where to look)

| Area | Path |
|------|------|
| JEPA definition | `backend/ml/jepa_model.py` |
| Training outline / dataset stub | `backend/ml/train.py` |
| AoU extract → tensors (notebook) | `backend/notebooks/aou_extraction_pipeline.ipynb` |
| Counterfactual orchestration | `backend/services/counterfactual.py` |
| Async pipeline | `backend/worker/tasks.py` |
| API | `backend/main.py`, `backend/schemas/` |
| Frontend | `frontend/src/` |

---

## 5. Open integration points (for new contributors)

- **Training:** Activate the full loop in `train.py`, align `num_features` / `num_interventions` with exported AoU manifests, and save checkpoints consumed by `settings.jepa_checkpoint_path`.
- **Inference:** Ensure intervention IDs used at inference **match** the embedding table used in training (`action_embedding` indices).
- **Causal engine** currently lists **demo interventions** in code; training cohorts may use **different** \(K\) and labels — registry and model checkpoint must stay **consistent**.

---

*This document reflects the codebase as of the last update; verify `config.py` and checkpoint paths for your environment.*
