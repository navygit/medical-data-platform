# Medical Data Platform

**Production-style data infrastructure for 3D medical multimodal datasets — ingestion, quality control, governance, versioning and lineage.**

[![CI](https://github.com/navygit/medical-data-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/navygit/medical-data-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-129%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## What this is

Three case studies (BraTS, MIMIC-CXR, LiTS) built as **one internal platform** rather than three disconnected notebooks, because that is how this work is actually done: shared ingestion, one QC engine, one release format, one governance model.

The thesis of the repository is that **the data product is the deliverable**. Model training is included where it proves the data product loads and trains — not as the point.

> **Run it right now, with no dataset download:**
> ```bash
> pip install -e . && make demo
> ```
> Every pipeline runs against a synthetic corpus that mirrors the real datasets' schema. See [Why synthetic data](#why-synthetic-data-and-why-that-is-the-point).

---

## Table of contents

- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Case study 1 — BraTS](#case-study-1--brats-3d-brain-mri-segmentation-pipeline)
- [Case study 2 — MIMIC-CXR](#case-study-2--mimic-cxr-multimodal-imagetext-pipeline)
- [Case study 3 — LiTS](#case-study-3--lits-data-governance-quality-and-bias)
- [Why synthetic data](#why-synthetic-data-and-why-that-is-the-point)
- [Engineering decisions](#engineering-decisions-worth-defending)
- [Honest limitations](#honest-limitations)
- [Project layout](#project-layout)
- [Development](#development)

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │      common/  (framework)    │
                        │  config · logging · metadata │
                        │  qc · storage · versioning   │
                        │  visualization               │
                        └───────────────┬──────────────┘
                                        │  every pipeline composes these
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
     ┌────────▼────────┐      ┌─────────▼────────┐      ┌─────────▼────────┐
     │  pipelines/     │      │  pipelines/      │      │  pipelines/      │
     │     brats       │      │     mimic        │      │      lits        │
     │                 │      │                  │      │                  │
     │ 3D segmentation │      │  image + text    │      │   governance     │
     │ data pipeline   │      │  fusion dataset  │      │   quality · bias │
     └────────┬────────┘      └─────────┬────────┘      └─────────┬────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │  Immutable, content-addressed │
                        │  releases + lineage + cards   │
                        └───────────────────────────────┘
```

**The data path is identical in all three**, which is the point of a platform:

```
Raw  →  Ingest  →  QC  →  Preprocess  →  Split  →  Release
        (hash,     (gate,   (normalise)   (group   (manifest,
         probe)     report)                by       hashes,
                                           patient) lineage)
```

---

## Quickstart

Requires Python 3.11 or 3.12. (Not 3.13/3.14 — MONAI has no wheels there yet.)

```bash
git clone https://github.com/navygit/medical-data-platform.git
cd medical-data-platform

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

make demo                          # synth data → all three pipelines
```

`make demo` takes a clean clone to a complete artifact set in about a minute on a laptop, with **no credentialed download and no GPU**. CI asserts this on every push.

### What you get

| Artifact | Path |
|---|---|
| QC report (self-contained HTML) | `outputs/brats/QC_REPORT.html` |
| Slice/mask/prediction overlays | `outputs/brats/figures/*.png` |
| Generated dataset card | `outputs/lits/dataset_card.md` |
| Bias audit with statistical tests | `outputs/lits/bias_audit.csv` |
| PHI audit | `outputs/mimic/PHI_AUDIT.csv` |
| Structured reports from free text | `data/processed/mimic/structured_reports.csv` |
| Versioned release + SHA-256 manifest | `releases/*/v1.0.0/manifest.json` |

### Individual pipelines

```bash
make brats        # or: python -m pipelines.brats.run --config configs/brats.yaml
make mimic
make lits

# Any config value is overridable from the CLI:
python -m pipelines.brats.run --set qc.min_tumor_voxels=50 --set split.seed=7
python -m pipelines.lits.run  --set extra.cohort=exploratory_all_liver_ct
```

### Optional: train the baseline

```bash
pip install -e ".[ml]"                       # adds torch + MONAI
python -m pipelines.brats.train --config configs/brats.yaml
python -m pipelines.brats.evaluate --split test
```

---

## Case study 1 — BraTS: 3D brain MRI segmentation pipeline

**Question: can this cohort be trusted enough to train on?**

```
Scan folders → Validate → Extract metadata → Manifest → QC gate
    → Normalise spacing/orientation/intensity → Patient-grouped split
    → Figures → Versioned release → (optional) 3D U-Net → Dice/IoU/HD95
```

### The QC engine, demonstrated

The synthetic generator **plants known defects on purpose**, so QC has real problems to find and the tests can assert it finds *exactly* those. From an actual run over 13 subjects:

| Subject | Planted defect | QC verdict |
|---|---|---|
| `BraTS2021_00003` | truncated NIfTI (corrupt gzip) | ❌ rejected — `corrupt_volume` |
| `BraTS2021_00005` | all-background mask | ❌ rejected — `empty_mask` |
| `BraTS2021_00007` | missing FLAIR | ❌ rejected — `missing_modalities` |
| `BraTS2021_00008` | zero-variance volume | ❌ rejected — `empty_volume` |
| `BraTS2021_00013` | byte-identical copy of `00001` | ❌ rejected as copy; `00001` retained |
| `BraTS2021_00006` | LAS orientation | ⚠️ warned → **reoriented**, kept |
| `BraTS2021_00009` | 6 mm slice thickness | ⚠️ warned → **resampled**, kept |

**8 of 13 subjects survived → 3/2/3 patient-grouped split → release `v1.0.0`.**

The error/warning distinction is the interesting part. A corrupt file is unusable, so it is an error. A non-standard orientation is *fixable*, so it is a warning and the pipeline fixes it. Treating both as fatal would silently throw away good data.

**Duplicate policy:** the first subject (sorted) is retained as canonical and later copies rejected. Failing both would discard the underlying study — the data is fine, the *duplication* is the defect. Only content hashing catches this; the files have different names and different patient IDs.

### Modules

| Module | Responsibility |
|---|---|
| `ingest.py` | Walk the tree, hash, probe geometry/intensity, build the manifest |
| `preprocess.py` | Reorient → resample → normalise intensity |
| `split.py` | Patient-grouped, tumour-burden-stratified partition with a leakage assertion |
| `train.py` | MONAI 3D U-Net baseline (manifest-driven loader) |
| `evaluate.py` | Dice, IoU, Hausdorff-95 per region + prediction figures |
| `run.py` | Orchestrator |

---

## Case study 2 — MIMIC-CXR: multimodal image/text pipeline

**Question: can images, reports and labels be linked without lying about the join?**

```
DICOM headers → PHI audit → Report parsing (sections, negation, severity)
   → Join on study_id → CheXpert-style labels → Patient timeline
   → Subject-grouped split → Fusion dataset → Release
```

### Report parsing is where this gets real

The naive approach — search for `"pneumonia"`, set the label — is **wrong about a third of the time**, because radiology prose is dominated by negation and hedging. This implements a NegEx/NegBio-style algorithm with bounded scope, sentence boundaries and termination phrases:

| Report text | Naive keyword | This parser |
|---|---|---|
| `No evidence of pneumonia.` | positive ❌ | **negative** ✅ |
| `Pneumonia cannot be excluded.` | positive ❌ | **uncertain** ✅ |
| `No pleural effusion or pneumothorax.` | both positive ❌ | **both negative** ✅ |
| `No effusion, but consolidation is present.` | both negative ❌ | **effusion −, pneumonia +** ✅ |
| `Heart size is normal. Small effusion.` | negative ❌ | **positive** ✅ |

That last row is a sentence-boundary case; the fourth is scope termination at *"but"*. All are covered by parametrised tests.

Labels follow **CheXpert semantics**: `1` positive / `0` negative / `-1` uncertain / `None` not mentioned. All four states are preserved in the dataset — collapsing uncertain into negative is a *modelling* decision, so `binarise_labels()` applies it as a **named policy** (`zeros` / `ones` / `ignore`) at training time rather than destroying information at ingest.

**Rule-based, deliberately.** BioClinicalBERT is the natural upgrade and the interface is built to swap for it — but the CheXpert labeller itself is rule-based, and a regex engine that runs in CI beats a 2 GB model that does not.

### Guards against the three classic multimodal failures

1. **Silent inner joins** — orphan images and orphan reports are counted and reported; a join rate below 95% logs a warning.
2. **Patient leakage** — one patient has many studies over years; the split is grouped by subject and asserted.
3. **Label leakage** — `INDICATION: history of pneumonia` is the *referral question*, not a finding. Labels are extracted from FINDINGS/IMPRESSION only. There is a test for this.

**PHI audit:** MIMIC-CXR ships de-identified, but a platform that *assumes* that and is wrong has caused a breach. Every identifier tag is re-checked on every file. Verifying is cheap; assuming is not.

---

## Case study 3 — LiTS: data governance, quality and bias

**Trains no model at all.** The deliverable is the governance artifact set — which is the Data Manager's actual product, and the part of the lifecycle that decides whether every downstream model is trustworthy.

```
CT + clinical metadata → Quality score (0–100) → Cohort build (attrition trail)
   → Bias audit (χ² / Kruskal-Wallis + effect sizes) → Dataset card → Release
```

### Graded quality scores, not pass/fail

Binary QC forces a false choice: slightly coarse spacing is not equivalent to a corrupt file. Each study scores 0–100 as a weighted sum, so the release policy picks a threshold per use case:

| Component | Weight | Penalises |
|---|---|---|
| integrity | 0.30 | unreadable / non-finite / constant |
| metadata completeness | 0.15 | missing demographics, acquisition params |
| spacing consistency | 0.15 | anisotropic or coarse voxels |
| noise | 0.15 | low SNR |
| contrast | 0.15 | poor dynamic range |
| slice continuity | 0.10 | dropped / duplicated slices |

Every component returns a score **and a human-readable reason**, so a rejected study can be explained to the clinician who submitted it.

### Cohorts are declarative, and attrition is the real output

```python
CohortSpec(name="adult_contrast_liver_ct", criteria=[
    Criterion("adult",             "age >= 18",            "Paediatric liver anatomy differs."),
    Criterion("contrast_enhanced", "contrast == True",     "Lesion conspicuity depends on contrast."),
    Criterion("has_label",         "has_label == True",    "Supervised training requires a mask."),
    Criterion("quality_threshold", "quality_score >= 65",  "Grade C or better."),
])
```

From a real run — this table is what makes a cohort *defensible*, and it is the first thing a reviewer asks for:

| step | criterion | n_before | n_after | n_removed |
|---|---|---|---|---|
| 0 | all_studies | 12 | 12 | 0 |
| 1 | adult | 12 | 12 | 0 |
| 2 | contrast_enhanced | 12 | 7 | **5** |
| 3 | has_label | 7 | 7 | 0 |
| 4 | quality_threshold | 7 | 6 | **1** |

A criterion that fails to evaluate **raises** rather than being skipped — a silently ignored inclusion rule produces a cohort that does not match its own definition.

### Bias auditing with statistics, not vibes

A bar chart showing 68% male tells you the number, not whether it matters. Every audit pairs a distribution with a **test and an effect size**:

- Representation vs. reference population → χ² goodness-of-fit + **Cramér's V**
- Subgroup outcome disparity → Kruskal-Wallis + **disparity ratio** (max/min median)

Effect size is reported alongside p-values on purpose: at 300k studies every trivial difference is "significant"; at 12 studies nothing is. **The disparity ratio is what a reviewer acts on.** χ² is skipped entirely when any expected cell < 5, rather than reporting a number that does not hold.

The audit correctly detects the skew planted in the generator (M at 67%, age 60–79 at 67%).

### Dataset card, generated

`outputs/lits/dataset_card.md` — ~170 lines, **every number traced to a run artifact**. A hand-written card describes what someone believed six months ago.

It includes the section most portfolios omit: **"Uses that are NOT recommended"**. A dataset card listing only strengths is marketing.

---

## Why synthetic data — and why that is the point

BraTS, MIMIC-CXR and LiTS are credentialed, multi-hundred-GB downloads. Code written against a specific download cannot be run by a reviewer who clones the repo, and cannot be tested in CI.

So the pipelines are written against a **schema**, and `scripts/generate_synthetic_data.py` emits data in that schema: real NIfTI volumes with real affines, real DICOM with real tags, real report prose. **The same code path runs on synthetic and real data** — point `paths.raw` at a real download and nothing else changes.

This is not a workaround. It buys three things that matter more than the pixels:

1. **The pipeline is executable by anyone**, including CI, on every push.
2. **QC is provably correct.** Defects are planted deliberately, so the tests assert that each check catches the defect it exists for — a guarantee you cannot get from real data, where you do not know the ground-truth defect list.
3. **No PHI ever touches the repo.**

To run against real data:

```bash
python -m pipelines.brats.run --set paths.raw=/data/BraTS2021_TrainingData
```

---

## Engineering decisions worth defending

**Splitting is grouped by patient and *asserted*, not assumed.** The most common defect in medical ML papers is splitting at the scan level when the unit of independence is the patient. `verify_no_leakage()` raises on every run. One patient with four modalities is four rows; a random row split puts the same brain in train and test and the Dice score becomes fiction.

**Ingest never raises on bad data.** A corrupt volume becomes a manifest row with null statistics. Ingest answers *"what is there?"*; QC answers *"is it usable?"*. Merging them gives you a pipeline that dies on the first bad file at 3am. A missing row silently shrinks the cohort; a null row is auditable.

**Labels resample with nearest-neighbour, never linear.** Linear interpolation of a label map invents classes that never existed. There is a test asserting the label set is unchanged after resampling.

**Z-score uses foreground only.** Background is ~70% of a brain MRI. Including it drags the mean toward zero and compresses foreground contrast. (CT uses fixed HU windows instead — HU *is* calibrated, so z-scoring would destroy real information. Hence `intensity_norm: none` in `configs/lits.yaml`.)

**Raw is never auto-created.** `paths.mkdirs()` deliberately skips the raw directory. Creating it would mask the most common misconfiguration there is: a typo'd path would silently ingest zero files and publish an empty release. Inputs are asserted, not created.

**QC checks are a registry.** Adding a rule is one decorated function — no edits to the engine. Clinical reviewers propose rules; a rule that lives in its own function with its own docstring can be reviewed by someone who does not read the whole pipeline. A check that raises becomes an ERROR *finding*, never a crash: one malformed study must not stop QC of the other 4,999.

**Releases are immutable and content-addressed.** `write_release` refuses to overwrite. The `dataset_hash` derives from sorted per-file hashes, so identical content yields an identical hash regardless of when or where it was built. `verify_release()` re-hashes against disk — that is what turns "we version our data" into something an auditor can trust.

**Manifests store relative POSIX paths.** Data is curated on Windows workstations and trained on Linux boxes. CI runs the matrix on both, because that is where path handling bites.

**Structured logging cannot crash the pipeline.** `extra={"message": ...}` raises `KeyError` in the stdlib — and QC findings legitimately *have* a message field. `SafeExtraAdapter` renames collisions instead of raising. (This was a real bug, found by running the code.)

---

## Honest limitations

Stated plainly, because a reviewer will find these anyway and the omission would say more than the limitation.

- **The trained model is a plumbing proof, not a result.** On the synthetic corpus it reaches Dice ≈ 0.02–0.23 — three subjects, three epochs, noise phantoms. It demonstrates the released dataset loads, trains and evaluates. Nothing more. Real BraTS with a real schedule is a different exercise.
- **Report parsing is rule-based** and will miss phrasings outside its vocabulary. BioClinicalBERT is the upgrade path; the interface is built for it.
- **De-identification is a subset** of the DICOM PS3.15 profile and does not include burned-in pixel text detection. Sufficient for auditing a de-identified archive; **not** sufficient to de-identify raw PACS data.
- **Quality-score weights are defensible, not empirical.** They encode judgement, which is why they live in one reviewable dict rather than scattered through the code. Calibrating them against radiologist ratings is the honest next step.
- **The SNR and contrast estimates are crude** — relative signals for ranking studies within a cohort, not physics-grade measurements.
- **Demographic fields are coarse** (binary sex, age bands) and cannot support intersectional fairness analysis.
- **Prefect, DVC, MLflow and Hydra are wired but optional.** The core runs without an orchestration server, because a reviewer reading the governance code should not have to stand up infrastructure. `dvc.yaml` and the Prefect flow wrap the same functions.

---

## Project layout

```
medical-data-platform/
├── common/                     # shared framework — dataset-agnostic
│   ├── config.py               # typed layered YAML + CLI overrides
│   ├── logging.py              # structured JSON/human logs
│   ├── metadata.py             # ScanRecord + manifest I/O (parquet/csv/json)
│   ├── qc.py                   # pluggable check registry + HTML/CSV reports
│   ├── storage.py              # filesystem access + SHA-256 hashing
│   ├── versioning.py           # immutable content-addressed releases + lineage
│   └── visualization.py        # headless PNG export
│
├── pipelines/
│   ├── brats/                  # ingest, preprocess, split, train, evaluate, run
│   ├── mimic/                  # dicom_parser, report_parser, fusion_dataset, run
│   └── lits/                   # quality_score, cohort_builder, bias_audit,
│                               #   dataset_card, run
│
├── configs/                    # base.yaml + one per case study
├── scripts/                    # synthetic data generator
├── tests/                      # 129 tests, 91% coverage
├── .github/workflows/ci.yml    # lint · types · tests (Linux+Windows) · e2e · docker
├── Dockerfile                  # multi-stage, non-root
├── Makefile                    # make help
└── pyproject.toml              # core / [ml] / [orchestration] / [dashboard] / [dev]
```

---

## Development

```bash
make help          # list all targets
make test          # full suite with coverage
make test-fast     # unit tests only (skip end-to-end)
make lint          # ruff
make format        # black + ruff --fix
make ci            # everything CI runs
```

### Test suite

**129 tests, 91% coverage.** The tests worth reading:

- `test_qc.py` — asserts each **planted defect** is caught by the check that exists for it. This is a behavioural contract, not an implementation detail: it is the reason to trust the pipeline.
- `test_pipelines.py::test_split_never_leaks_a_patient` — the invariant that makes every reported metric meaningful.
- `test_pipelines.py::test_negation_and_uncertainty` — parametrised over the adversarial report phrasings above.
- `test_pipelines.py::test_resample_nearest_preserves_label_values` — asserts interpolation never invents a class.
- `test_end_to_end.py::test_brats_release_detects_corruption_after_the_fact` — mutates a released file and asserts verification catches it.

I/O is **not mocked**. Tests run against real NIfTI and DICOM files on disk, because the bugs that actually occur in medical data pipelines *are* I/O bugs: a truncated gzip, an unexpected affine, a tag that is present but empty. Mocking nibabel would test the mock.

### Tech stack

`Python 3.11+` · `pydantic` · `pandas` · `pyarrow` · `nibabel` · `pydicom` · `numpy` · `scipy` · `scikit-learn` · `matplotlib` · `plotly`
Optional: `torch` · `MONAI` · `MLflow` · `Prefect` · `DVC` · `Hydra` · `Polars` · `Dash`

---

## Contact

**Navin Kumar** — [LinkedIn](https://www.linkedin.com/in/navin91/) · [GitHub](https://github.com/navygit)

Built as a demonstration of end-to-end medical imaging data operations: ingestion, quality control, governance, versioning and lineage. Questions about the design decisions are welcome — open an [issue](https://github.com/navygit/medical-data-platform/issues).

---

## Licence

MIT — see [LICENSE](LICENSE). The underlying datasets retain their own licences and access terms; nothing here redistributes them.
