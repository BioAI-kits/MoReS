# MoReS

Multi-omics integration can improve survival prediction in bladder cancer, but the benefits of joint models may be limited when sample sizes are small or inputs are incomplete or perturbed. We developed MoReS to estimate the predictive gain of joint modeling for each patient and adjust the degree of fusion while accounting for censoring and estimation uncertainty.

## Data availability

The datasets described in the manuscript were obtained from previously published research resources:

| Cohort | Source and accession | How to obtain the data |
|---|---|---|
| TCGA-BLCA, TCGA-BRCA, TCGA-KIRC, and TCGA-LUAD | [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/) | Select the corresponding TCGA project identifier to obtain molecular data and clinical records. Follow the access conditions specified for the requested files. |
| GSE13507 bladder cancer cohort | [NCBI Gene Expression Omnibus: GSE13507](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13507) | Obtain expression data, sample annotations, and available supplementary files from the GEO series record. |
| Danish muscle-invasive bladder cancer (MIBC) cohort | [Taber et al. (2020)](https://doi.org/10.1038/s41467-020-18640-0); EGA study [EGAS00001004505](https://ega-archive.org/studies/EGAS00001004505), dataset [EGAD00001006238](https://ega-archive.org/datasets/EGAD00001006238) | Consult the publication for cohort information. Raw RNA-seq data are under controlled access. Request access through the EGA dataset record and follow its download instructions after authorization. |

Patient data are not distributed with this repository. Prepare molecular measurements and clinical outcomes in one of the formats described under [Data preparation](#data-preparation). The `prepare-xena` command reads processed TCGA matrices in Xena format; `import-tables` reads patient-by-feature CSV files. Raw sequencing files require upstream processing before use with these interfaces.

## Installation

### Requirements

- Python **3.11 or later**.
- CPU execution is supported. NVIDIA GPU training requires a compatible driver and a CUDA-enabled PyTorch installation.
- Dependencies are declared in `pyproject.toml` and installed automatically.

| Dependency | Minimum version | Purpose |
|---|---|---|
| NumPy | 1.26 | Numerical calculations |
| pandas | 2.1 | Tabular data and patient alignment |
| SciPy | 1.11 | Optimization and numerical utilities |
| scikit-learn | 1.4 | Data splitting, preprocessing, and ridge regression |
| scikit-survival | 0.23 | Cox and LASSO-Cox survival models |
| PyTorch | 2.2 | Neural network training |
| Joblib | 1.3 | Model serialization |

### Clone the repository

```bash
git clone git@github.com:BioAI-kits/MoReS.git
cd MoReS
```

If the repository is already available locally, enter its existing root directory.

### Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install the package and dependencies

```bash
python -m pip install -e .
```

Editable installation (`-e`) makes source changes available without reinstalling the package. Run this command from the repository root containing `pyproject.toml`.

For CUDA, select the appropriate installation command from the [official PyTorch installation page](https://pytorch.org/get-started/locally/), then install this package. Check GPU availability with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The supplied configurations use CPU execution. To train on a GPU, set `device` to `"cuda"` or `"cuda:0"` in your configuration. LASSO-Cox and ridge regression still run on the CPU; the saved-model prediction interface uses the CPU.

### Check the installation

```bash
python main.py --help
python -m pip check
```

Dependencies use minimum-version constraints; major library versions are recorded for training runs.

## Quick start

Run the examples from the repository root. Commands use macOS / Linux shell syntax; on Windows, combine multiline commands into a single line. Replace `/path/to/...` with actual paths and retain quotes around paths containing spaces. Relative paths are resolved from the working directory.

This example uses synthetic data and a reduced configuration to exercise the workflow. Its outputs are not scientific results.

```bash
# 1. Generate 512 synthetic patients with 6 features per molecular modality
python main.py make-demo --patients 512 --features 6 --seed 7 --output runs/quickstart-data

# 2. Inspect patient counts, feature counts, and modality availability
python main.py audit --data runs/quickstart-data

# 3. Run a small nested cross-validation workflow
python main.py cv --data runs/quickstart-data --config configs/smoke.json --output runs/quickstart-cv

# 4. Train and save a model
python main.py train --data runs/quickstart-data --config configs/smoke.json --output runs/quickstart-model

# 5. Load the saved model and generate predictions
python main.py predict --data runs/quickstart-data --model runs/quickstart-model/model.joblib --times 0 6 12 24 36 60 --output runs/quickstart-predictions
```

Step 5 uses the training data to demonstrate prediction. Use the out-of-fold (OOF) predictions from step 3 to assess internal generalization. Commands that create output directories require a new directory; choose a different `--output` path when repeating a run.

## Data preparation

### Clinical outcomes and covariates

Provide one row per patient in a CSV file:

```csv
patient_id,os_time_months,os_event,age,sex,stage
TCGA-XX-0001,24.5,1,68,male,Stage III
TCGA-XX-0002,36.0,0,61,female,
```

These rows illustrate the format only.

| Field | Definition |
|---|---|
| `patient_id` | Unique patient identifier matching the molecular data |
| `os_time_months` | Original OS follow-up time in months; must be finite and positive |
| `os_event` | `1` for death and `0` for right censoring |
| `age` | Age in years |
| `sex` | `male` or `female`; leave missing values empty |
| `stage` | `Stage I` through `Stage IV`; letter suffixes for substages are supported; leave missing values empty |

Training and evaluation require the outcome fields. Prediction-only data may omit `os_time_months` and `os_event`, but must include the patient ID and clinical covariate columns. Covariate values may be missing. Retain original follow-up times; administrative truncation is applied separately during training.

To extract clinical variables and outcomes from TCGA Xena files:

```bash
python main.py extract-clinical \
  --xena-dir "/path/to/data/tcga_xena/BLCA" \
  --cancer BLCA \
  --output data/blca-clinical.csv
```

The extractor converts OS from days to months by dividing by `30.4375`. It writes valid records to the requested CSV and invalid-outcome records to a corresponding `.excluded.csv` file. Define the patient inclusion criteria for your study before analysis.

### TCGA matrices in Xena format

For BLCA, the directory passed to `--xena-dir` must contain:

```text
TCGA-BLCA.star_tpm.tsv.gz
TCGA-BLCA.methylation450.tsv.gz
TCGA-BLCA.gene-level_ascat3.tsv.gz
TCGA-BLCA.somaticmutation_wxs.tsv.gz
```

The `extract-clinical` command additionally requires `TCGA-BLCA.clinical.tsv.gz` and `TCGA-BLCA.survival.tsv.gz` in that directory. These are processed table formats; renaming arbitrary GDC downloads does not convert them into compatible inputs. For other processed layouts, use the CSV import interface below.

```bash
python main.py prepare-xena \
  --clinical data/blca-clinical.csv \
  --xena-dir "/path/to/data/tcga_xena/BLCA" \
  --cancer BLCA \
  --rna-source-scale log2p1 \
  --output data/blca

python main.py audit --data data/blca
```

Specify the RNA input scale explicitly:

| `--rna-source-scale` | Input | Processing |
|---|---|---|
| `raw` | Nonnegative TPM values | Apply `log2(TPM + 1)` |
| `log2p1` | Values already transformed as `log2(TPM + 1)` | Keep the supplied values |

Do not log-transform an already transformed expression matrix. Methylation inputs must be beta values in `[0, 1]`; CNV retains the source numerical scale. The importer selects primary-tumor samples, prioritizes 01A samples, and then uses sample-ID ordering to resolve remaining choices. Sample mappings are saved with the prepared data.

Mutations are encoded as binary gene-level status. By default, modality availability is inferred from matching mutation records. If assay coverage is known independently, provide `--mutation-coverage /path/to/coverage.csv` with columns `patient_id,available` to distinguish assayed patients without reported variants from patients without mutation measurements.

Large matrices are processed in chunks. `--feature-limit 32` retains only the first 32 source features for an input check; it is not a feature-selection procedure. Imputation, standardization, and LASSO-Cox selection are fitted within each training partition.

### Custom or external cohorts

Each molecular CSV uses patients as rows and features as columns:

```csv
patient_id,feature_1,feature_2
patient_001,1.2,3.4
patient_002,,2.1
```

```bash
python main.py import-tables \
  --clinical "/path/to/clinical.csv" \
  --rna "/path/to/rna.csv" \
  --methylation "/path/to/methylation.csv" \
  --cnv "/path/to/cnv.csv" \
  --mutation "/path/to/mutation.csv" \
  --output data/custom-cohort
```

Omit arguments for unavailable molecular modalities. This interface does not automatically log-transform RNA; supply values on the scale expected by the training configuration. Without `--availability`, modality availability is inferred from finite observations. To specify availability explicitly, pass a binary CSV with columns `patient_id,rna,methylation,cnv,mutation,clinical`.

Align external feature identifiers with the training data before import. Use documented annotation and duplicate-handling rules when mapping Ensembl IDs, gene symbols, or probe IDs. RNA-only patients use the available experts in the reference pathway. The joint pathway is activated only when at least three molecular modalities are available.

### Prepared data and missing values

A prepared dataset contains `clinical.csv`, `availability.csv`, `manifest.json`, and a `.dat` matrix plus `.features.json` file for each molecular modality. Matrices use float32 feature-major storage and read-only memory maps. The manifest fixes patient order. Patient IDs, outcome coding, follow-up times, feature identifiers, and binary mutation values are checked during preparation or loading.

NaN represents an unobserved feature; a measured zero remains an observation. Entirely absent modalities have separate availability flags. Prediction aligns features by identifier and uses training-fitted imputation and scaling. TCGA preparation preserves source Ensembl versions and probe IDs; it does not automatically map external identifiers. Constant or entirely missing features are removed within training partitions.

Mutation tables contain called variants and may not list all assayed patients. Within patients marked as mutation-available, unlisted genes are coded zero. Use an independent assay-coverage table when needed; absence of a variant record alone does not establish whether a patient was assayed. All called variants contribute to binary gene status without an additional nonsynonymous-only filter.

Preparation records source paths, file sizes, modification times, the clinical-file SHA-256, and sample mappings. Model fitting materializes training subsets for feature selection, so memory requirements can exceed the size of the memory-mapped cache.

## Training and prediction

### Nested cross-validation

```bash
python main.py cv \
  --data data/blca \
  --config configs/method.json \
  --output runs/blca-cv
```

Repeated nested cross-validation operates at the patient level. Preprocessing, feature selection, reference weighting, gain learning, and fusion-parameter selection are restricted to the corresponding training partitions. Metrics are computed from pooled outer-fold predictions within each repetition, then summarized across repetitions.

### Train a final model

```bash
python main.py train \
  --data data/blca \
  --config configs/method.json \
  --output runs/blca-final
```

This command performs internal parameter selection and trains the final model on the supplied cohort. The main output is `runs/blca-final/model.joblib`. Use this model for new-patient prediction and cross-validation outputs for internal performance assessment.

### Predict new patients

First prepare new-patient data with `import-tables` or `prepare-xena`, then run:

```bash
python main.py predict \
  --data data/new-patients \
  --model runs/blca-final/model.joblib \
  --times 6 12 24 36 60 \
  --output runs/new-predictions
```

Prediction times are in months and must fall within the trained prediction window. If `--times` is omitted, the configured time boundaries are used. Prediction reuses the features, imputation values, and scaling parameters fitted during training. Outputs include reference, joint, and final survival curves, together with fusion diagnostics.

### Evaluate an independent cohort

```bash
python main.py evaluate \
  --data data/external \
  --model runs/blca-final/model.joblib \
  --horizon 36 \
  --output runs/external-36m
```

Evaluation requires OS outcomes and patient IDs disjoint from the training cohort. The interface uses the frozen training censoring model without refitting it on the target cohort.

To evaluate an input perturbation:

```bash
python main.py evaluate \
  --data data/external \
  --model runs/blca-final/model.joblib \
  --horizon 36 \
  --perturbation cnv_rescale \
  --severity 0.35 \
  --output runs/external-cnv
```

Supported perturbations are `rna_mask`, `methylation_noise`, `cnv_rescale`, `mutation_missing`, `modality_drop`, and `platform_shift`. Perturbations affect model inputs; outcomes and the original covariates used by the censoring model remain fixed.

For independent within-cancer retraining, prepare each cancer cohort separately and run `cv` or `train`. Setting `evaluate --horizon 36` changes the evaluation window only. To train a 36-month model, set `cuts` to `[0, 6, 12, 24, 36]` in a separate configuration.

## Configuration

| Setting | `configs/method.json` | `configs/smoke.json` |
|---|---|---|
| Purpose | Method training configuration | Reduced workflow check |
| Outer folds (`outer_folds`) | 5 | 2 |
| Repetitions (`repeats`) | 20 | 1 |
| Folds at internal splitting levels | 3 | 2 |
| Ensemble members (`ensemble_members`) | 3 | 1 |
| Gain bootstrap replicates | 100 | 5 |
| Maximum selected features per modality (`top_features`) | 50 | 4 |
| Network search trials (`search_trials`) | 12 | 1; uses the supplied network parameters |
| Maximum epochs | 200 | 2 |
| Censoring model (`censoring`) | `cox` | `km` |
| Device | `cpu` | `cpu` |

Other settings include `seed`, `threads`, `batch_size`, `hidden_dim`, `learning_rate`, `dropout`, `weight_decay`, `ridge_alphas`, `kappa_grid`, `eta_grid`, and `augmentation_beta`. See the configuration files and `src/mores/config.py` for all supported fields. Unknown fields are rejected.

Copy a configuration before changing it:

```bash
cp configs/method.json configs/my-run.local.json
# Edit my-run.local.json, for example to change device and threads
python main.py train --data data/blca --config configs/my-run.local.json --output runs/my-run
```

`max_candidates: null` leaves the variance-based candidate count uncapped. Setting a cap reduces the number of inputs to LASSO-Cox and changes the feature-selection procedure; record such changes in your analysis. Full-feature methylation data can require substantial memory, and the complete configuration involves multiple nested fitting stages.

### Implementation conventions

The reference pathway combines four molecular experts and one clinical expert using time-constant weights fitted from training-internal OOF predictions. The joint pathway uses a two-head attention block with masked mean pooling and is enabled only for patients with at least three molecular modalities. Clinical inputs do not count toward this threshold. Both pathways predict positive piecewise-constant hazards; events exactly on a time boundary belong to the interval ending at that boundary.

For pathway curves `S0` and `S1`, curve distance is `q = integral((S1-S0)^2) / tau`. Gain estimates are projected onto `[-sqrt(q), sqrt(q)]`. With bootstrap mean `g_bar` and population standard deviation `u`, fusion uses `ell = max(-sqrt(q), g_bar-kappa*u)` and `alpha = clip(ell/(q+eta), 0, 1)`. If `q=0` or the joint pathway is ineligible, `alpha=0`. Bootstrap variability describes gain-estimation instability with upstream predictions fixed; it is not a full-pipeline standard error or a calibrated confidence interval.

Some implementation details are explicit repository choices where the manuscript does not specify a complete procedure:

| Component | Implementation choice |
|---|---|
| Clinical encoding | Standardized age and an observation flag; sex and stage use one-hot encoding with explicit missing categories. Entirely missing clinical inputs have no clinical-expert contribution. |
| Feature selection | Training-local LASSO-Cox tuning by held-out Harrell concordance. Retain up to 50 nonzero features; do not pad shorter selections. Zero selected features permit an intercept-only expert. Nonconverged or numerically invalid candidates are rejected. |
| Network selection | Up to 12 random configurations, selected using mean reference and joint validation IBS. Early-stopping preprocessing excludes the stopping-validation patients; transforms and networks are subsequently refitted on the current training partition. |
| Reference weighting | IPCW IBS plus `0.01` times squared expert logits, using training-internal OOF expert predictions. |
| Gain fitting | Weighted ridge with an unpenalized intercept; patient-grouped penalty selection and bootstrap. Summary scaling is refitted within each ridge split and bootstrap; each patient's augmented records have total weight one. |
| Censoring | Ridge Cox with age, sex, stage, and modality masks, fitted on original follow-up and modality states; reverse Kaplan–Meier is configurable. All-constant covariates reduce to reverse Kaplan–Meier. |
| Follow-up support | Check raw censoring survival and numbers at risk before denominator clipping. Unsupported training windows fail explicitly; unsupported evaluation metrics remain missing and carry support status. |
| Integration | Gauss–Legendre quadrature segmented at monthly, hazard, event, and censoring boundaries. Curve distance uses a separate outcome-independent integration grid. |
| Concordance | `ipcw_cindex` uses conditional censoring weights. Marginal Uno concordance uses a separate reverse Kaplan–Meier estimate; the two are reported separately. |

All preprocessing and model selection remain inside the relevant training partitions, including network tuning, stacking, gain cross-fitting, and fusion selection. The `smoke.json` settings reduce computation for the quick-start example and are not an analysis protocol.

## Outputs

| Command or stage | File | Contents |
|---|---|---|
| Data preparation | `clinical.csv`, `availability.csv` | Aligned clinical records and modality availability |
| Data preparation | `*.dat`, `*.features.json`, `manifest.json` | Feature-major matrices, feature identifiers, patient order, and source metadata |
| `cv` | `metrics.csv` | Metrics and estimability status by repetition and strategy |
| `cv` | `summary.csv` | Means, standard deviations, and valid counts across repetitions |
| `cv` | `oof-*.npz` | Patient IDs, fold assignments, curves, integration grid, support flags, and gain/fusion diagnostics |
| `cv` | `calibration.csv` | Grouped predictions and IPCW-observed survival probabilities |
| `cv` | `repeat-*-fold-*.json` | Training splits, selected features, and parameter-selection records |
| `train` | `model.joblib` | Fitted model and preprocessing transforms |
| `train` | `provenance.json` | Final-model training provenance |
| `train` / `cv` | `config.json`, `runtime.json`, `data_manifest.json` | Configuration, runtime environment, and input provenance |
| `predict` | `predictions.npz`, `diagnostics.csv` | Survival curves and patient-level fusion diagnostics |
| `evaluate` | `metrics.csv`, `predictions.npz`, `evaluation.json` | External metrics, predictions, and evaluation settings |

Read a prediction file with NumPy:

```python
import numpy as np

with np.load("runs/new-predictions/predictions.npz", allow_pickle=False) as result:
    patient_ids = result["patient_ids"]
    months = result["times"]
    survival = result["mores"]  # Shape: patients x time points
    alpha = result["alpha"]    # One time-constant fusion weight per patient
    print(patient_ids.shape, months, survival.shape, alpha.shape)
```

Load Joblib model files only from trusted sources. Data directories, run outputs, and model binaries are excluded by `.gitignore`.

## License

This repository includes the Apache License 2.0. See [LICENSE](LICENSE).
