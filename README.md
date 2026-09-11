# A Cost-Aware Machine Learning Framework for Maintenance Optimisation on Standardised SAP PM Data

Failure prediction, cost-optimal preventive intervals and economically ranked maintenance
backlogs, built from **standard SAP Plant Maintenance transactional history alone** — no sensor
or IoT data required.

This repository contains the complete analysis pipeline for an MSc Data Science dissertation
(COM7014 Advanced Computing Project, Arden University), together with a fully synthetic,
ISO 14224-aligned SAP PM benchmark dataset whose generating parameters are disclosed.

---

## Why this exists

Predictive-maintenance research is dominated by condition monitoring: vibration, thermography,
dense IoT sensing. That paradigm presupposes instrumentation most operating plants do not have,
and it excludes precisely the mid-criticality asset base where interval misallocation is most
widespread. Meanwhile the failure, cost and work-execution history already sitting in SAP PM
tables — QMEL, AUFK, AFRU, MPLA — goes largely unused for decision-grade optimisation.

The framework asks whether that transactional residue is enough on its own. Four layers:

1. **Data** — a parameterised generator producing referentially intact SAP PM history
2. **Prediction** — per-class survival models estimated under censoring
3. **Decision** — cost-optimal preventive intervals and a cost-of-delay backlog ranking
4. **Evaluation** — counterfactual replay against the policies an organisation actually runs

Because the benchmark is synthetic with **disclosed ground truth**, every claim is testable at
two levels: does the model recover the known generating parameters, and does being right
translate into better decisions? Neither check is available to studies on proprietary data.

---

## Headline results

Pre-registered criteria, fixed before any model was fitted, reported without adjustment.

| Question | Criterion | Result | |
|---|---|---|---|
| **RQ1** Failure prediction | C-index ≥ 0.70 | **0.707** (Cox PH, temporal hold-out) | met |
| | Weibull shape recovery ≤ 15% | 16.4% median | missed |
| | Weibull scale recovery ≤ 15% | 9.3–27.1%, conditional on ρ | conditional |
| **RQ2** Interval optimisation | ≥ 15% cost-rate reduction | **1.8%** in replay | missed |
| **RQ3** Backlog prioritisation | ≥ 20% cost-of-delay reduction | **36.6%** | met |
| **RQ4** Combined framework | positive fleet-wide delta | **2.6% ± 1.4%** total cost | met |

Three findings are worth more than the numbers:

- **The restoration factor ρ is not identifiable** from transactional history. Three independent
  strategies — profile likelihood, concordance scoring, held-out likelihood — returned flat or
  boundary answers. Scale recovery is conditional on it; discrimination is not. Reported as a
  sensitivity band rather than a point estimate.
- **Parameter error and decision error decouple.** Cost-rate curves are flat near their minima,
  so interval error is cheap — which explains both why calendar maintenance persists in practice
  and why the attainable headroom here was ~8%, below the pre-registered criterion.
- **Prioritisation wins by preventing the *expensive* escalations, not more of them.** The 36.6%
  cost reduction came with escalation counts essentially unchanged (43 → 41).

---

## Repository layout

```
01_config.ipynb              shared constants: catalog, schema, volume, temporal split
02_setup.ipynb               creates schema, volume and folder scaffold (idempotent)
03_generate_dataset.ipynb    synthetic SAP PM generator -> volume landing zone
04_bronze_ingest.ipynb       raw entity extracts -> bronze Delta tables
05_silver_conform.ipynb      typing, conforming, referential integrity
06_gold_layer.ipynb          survival intervals, cost params, cycles, escalation set
07_quality_and_exports.ipynb data contract tests T1/T7/T8 + result exports
08_charts.ipynb              dataset and cost figures
09_ml_prediction_layer.ipynb survival model ladder, RQ1 evaluation
10_interval_optimizer.ipynb  cost-optimal intervals, regret vs truth (RQ2)
11_backlog_prioritizer.ipynb escalation model, cost-of-delay ranking (RQ3)
12_backtest.ipynb            four-regime counterfactual replay (RQ4)
```

**Filename order is execution order.** Each notebook includes the shared configuration via
`%run ./01_config`.

---

## Running it

### Requirements

- Databricks workspace with Unity Catalog enabled
- Permission to create a schema and volume in the target catalog
- `lifelines` and `xgboost` — needed by notebook 09 only

Everything else (pandas, numpy, scipy, matplotlib, scikit-learn) is in the standard runtime.
The generator and all estimation code are pure Python running on the driver; no cluster
scaling, GPU or distributed compute is required. Spark is used for Delta table management and
SQL transformations, not for scale — the dataset is deliberately modest.

### Setup

Edit `01_config.ipynb`:

```python
CATALOG = "your_catalog"
SCHEMA  = "sap_pm"
VOLUME  = "data_volume"
```

Grant the running identity:

```sql
GRANT USE CATALOG, CREATE SCHEMA ON CATALOG your_catalog TO `<principal>`;
GRANT USE SCHEMA, CREATE TABLE, CREATE VOLUME, MODIFY, SELECT
  ON SCHEMA your_catalog.sap_pm TO `<principal>`;
GRANT READ VOLUME, WRITE VOLUME
  ON VOLUME your_catalog.sap_pm.data_volume TO `<principal>`;
```

### Interactively

Run notebooks 01 to 12 in order. Set the `dataset_version` widget (e.g. `v2_2`) on **every**
notebook — widget values do not carry between notebooks.

### As a workflow

Create a job with one task per notebook, sourced from this repository. Dependencies:

```
02 → 03 → 04 → 05 → 06 → ┬→ 07
                         ├→ 08
                         ├→ 11
                         └→ 09 → 10 → 12
```

Set `dataset_version` as a **job parameter** rather than per-task widgets, so all tasks resolve
one value. Declare `lifelines` and `xgboost` in the environment for task 09; if you do, remove
the `%pip` and `%restart_python` cells from that notebook, because the restart clears the
session and discards the configuration loaded by `%run`.

End-to-end runtime is roughly ten minutes on serverless compute.

---

## Data architecture

Medallion pattern on Delta Lake under Unity Catalog.

```
generator  →  landing/*.csv
                   ↓
              bronze_*   raw, string-typed, provenance stamped
                   ↓
              silver_*   typed, conformed, integrity asserted
                   ↓
     ┌─────────────┬─────────────┬─────────────┐
gold_survival  gold_class_  gold_incumbent  gold_escalation
 _intervals    cost_params    _cycles          _training
     ↓             ↓             ↓                ↓
 RQ1 models   RQ2 optimizer  RQ4 baseline   RQ3 prioritizer
```

Two deliberate design points:

**Ground truth lives in a separate folder from the landing zone**, and nothing in silver or gold
joins to it. The two-level validation design only holds if the fitted models never see the
generating parameters; separate directories make leakage through a wildcard read structurally
impossible rather than merely unlikely.

**Preventive-terminated intervals are right-censored, not discarded.** They carry the
information that the item survived to that age, and dropping them biases lifetime estimates
downward. Where a preventive order and a failure fall on the same date, the preventive event is
ordered first — the pair reads as "maintained, then failed", which censors conservatively.

---

## The benchmark dataset

Ten ISO 14224-aligned equipment classes, 550 items, six years of history across seven SAP
entities (IFLOT, EQUI, MPLA, QMEL, AUFK, AFRU, COSS).

Failure behaviour is a **Weibull renewal process with imperfect preventive maintenance**: each
PM applies a virtual-age restoration factor ρ in the sense of Kijima, so maintenance policy has
a genuine causal effect on failure incidence and cost-optimal intervals exist to be found.
Hazard varies by manufacturer, plant operating context and criticality — observable covariates a
model can learn — plus a residual frailty term it cannot. A defect-escalation overlay generates
minor notifications that become breakdowns if unaddressed, with the originating-notification
linkage serving as ground truth for the prioritiser.

Disclosed in `ground_truth/<version>/`:

| File | Contents |
|---|---|
| `simulation_parameters.csv` | per-class Weibull β and η, ρ, cycles, cost structure |
| `equipment_frailty.csv` | per-item frailty and effective characteristic life |
| `covariate_effects.csv` | manufacturer, plant and criticality hazard effects |

The generator is seed-deterministic: a fixed seed reproduces the dataset byte-for-byte.

**No personal data.** Records are fully synthetic; the personnel field holds generated `TECH-n`
tokens, verified at value level by the test suite rather than assumed.

---

## Reproducibility

- Seed-deterministic generation
- Every bronze row stamped with its dataset version and ingest timestamp
- Every load recorded in an append-only `_ingest_manifest`
- Delta transaction log binding each model run to an immutable data snapshot
- Data contract, standards-alignment and no-personal-data tests executed as code into
  `dq_results`, not asserted in prose

All pipeline tables are written with overwrite semantics, so a re-run replaces rather than
accumulates. The one exception is `_ingest_manifest`, which appends by design.

---

## Known limitations

Stated plainly, because the results should be read against them.

- **Synthetic-data validity boundary.** Results demonstrate what the framework can do on data
  whose generating process belongs to the model family it assumes. This establishes
  methodological capability, not field-proven savings.
- **ρ is unidentifiable** from transactional history — an irreducible dependence on an
  unverifiable maintenance-effectiveness assumption for anyone fitting lifetime distributions to
  CMMS exports.
- **The escalation overlay is independent** of the equipment's hazard state, whereas real
  deteriorating equipment plausibly generates both more defects and faster escalation.
- **Maintenance plans are single-cycle**; real MPLA usage includes strategy plans and packages.
- **No seasonality or operating-context variation** beyond the plant-level term.
- **Data is clean.** Real work-order data is not; robustness to missing damage codes,
  misclassified order types and free-text-only symptoms is untested.
- **The cost objective is indifferent to consequence.** It would extend intervals on
  safety-critical equipment if the arithmetic favoured it. Operational use needs an explicit
  safety-criticality override.

---

## Citation

```
Bhat, A. (2026) A Cost-Aware Machine Learning Framework for Maintenance Optimisation on
Standardised Plant Maintenance Data: Failure Prediction, Interval Optimization, and Work
Prioritization. MSc dissertation, Arden University.
```

## Licence

Code: MIT. Dataset and generated outputs: CC-BY 4.0.
