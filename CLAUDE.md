# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic fraud detection project for **PLUS TI – Universidad del Valle 2025**. The goal is binary classification of bank transactions using ISO 8583-structured synthetic data, with two main components:

- **Part A**: Optimize detection of fraud in lodging/hospitality merchants (MCC 3501–3999 and 7011) using LightGBM with custom `feval` metrics to minimize false positives.
- **Part B**: Federated learning across three banks (BO-VIP, BR-PRIVATE, GT-STATE) — train on banks 1 & 2, infer on bank 3 (which has no `is_fraud` label).

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run notebooks:

```bash
jupyter notebook
```

## Data Architecture

Raw datasets live in `data/raw/` — three CSVs (~100k rows each) separated by semicolons (`;`), one per bank:

| File | Bank | Country | Tier |
|---|---|---|---|
| `01_bo_vip_seed22_n100000.csv` | BO-VIP | Bolivia | VIP |
| `02_br_privado_seed33_n100000.csv` | BR-PRIVATE | Brazil | Private |
| `03_gt_estatal_seed3_n100000.csv` | GT-STATE | Guatemala | State |

Bank 3 (`03_gt_estatal`) has no `is_fraud` column — it is inference-only.

Processed data lives in `data/processed/`:
- `eda_base_dataset.csv` — cleaned dataset with initial derived features (output of `01_eda.ipynb`)
- `train_dataset.csv` — months 1–5 (temporal split)
- `test_dataset_june.csv` — month 6 only (held-out test set)

## Notebook Pipeline

Notebooks must be executed in order:

1. `01_eda.ipynb` — EDA, data quality, temporal split into train/test
2. `02_feature_engineering.ipynb` — Feature engineering (stub)
3. `03_baseline_model.ipynb` — LightGBM baseline (stub)
4. `04_custom_metrics.ipynb` — Custom `feval` functions (stub)
5. `05_optimization.ipynb` — Hyperparameter tuning with Optuna (stub)
6. `06_federated_model.ipynb` — Federated training across banks (stub)

Notebooks 02–06 are currently empty and need to be implemented.

## Key Domain Rules

**Target variable**: `is_fraud` (bool) — ~5% positive rate (severely imbalanced).

**Temporal split**: Test set is month 6 (`transaction_month == 6`). Never use future data in training — respect chronological order to prevent data leakage.

**Data leakage risk**: Variables `approved`, `DE39_response_code`, `response_description`, and `DE38_authorization_code` reflect post-authorization decisions and must be excluded from model features.

**Lodging MCC filter**:
```python
hotel_mccs = list(range(3501, 4000)) + [7011]
df["is_hotel"] = df["DE18_merchant_category_code"].isin(hotel_mccs)
```

**Key predictive features** (from EDA correlation analysis):
- `amount_baseline_ratio` = `amount_usd / client_baseline_amount`
- `amount_vs_baseline` = `amount_usd - client_baseline_amount`
- `amount_usd`, `is_international`, `distance_from_home_km`
- `DE22_pos_entry_mode`, `DE25_pos_condition_code`, `channel`
- `DE18_merchant_category_code`

**Primary metric**: FP Ratio = FP / (TP + FP) — minimize while maintaining recall. Do not use accuracy.

**Custom feval target** (conceptual):
```python
score = hotel_recall - alpha * false_positive_rate
```

## ISO 8583 Field Naming Convention

Raw columns prefixed with `DE` + number follow the ISO 8583 Data Element numbering (e.g., `DE4_amount_transaction`, `DE18_merchant_category_code`, `DE22_pos_entry_mode`). The `docs/` directory contains the full variable descriptions.
