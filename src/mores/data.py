"""Patient-aligned raw data. No fitted transform is applied during ingestion."""
from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import hashlib
import numpy as np
import pandas as pd

MODALITIES = ("rna", "methylation", "cnv", "mutation")


@dataclass
class Cohort:
    ids: np.ndarray
    clinical: pd.DataFrame
    x: dict[str, np.ndarray]  # patients x raw features; may be memory mapped
    features: dict[str, list[str]]
    available: np.ndarray  # four molecular modalities, then clinical
    time: np.ndarray | None = None
    event: np.ndarray | None = None

    def __len__(self):
        return len(self.ids)

    def subset(self, indices):
        ix = np.asarray(indices)
        return Cohort(self.ids[ix], self.clinical.iloc[ix].reset_index(drop=True),
                      {m: a[ix] for m, a in self.x.items()}, self.features,
                      self.available[ix], None if self.time is None else self.time[ix],
                      None if self.event is None else self.event[ix])

    def validate(self, outcomes=False):
        n = len(self)
        if n == 0 or len(set(self.ids)) != n:
            raise ValueError("Cohort requires nonempty unique patient IDs")
        if len(self.clinical) != n or not {"age", "sex", "stage"} <= set(self.clinical):
            raise ValueError("Clinical table must align with IDs and include age/sex/stage")
        if self.available.shape != (n, 5) or not np.isin(self.available, [0, 1]).all():
            raise ValueError("availability must be a binary n x 5 array")
        if not self.available.any(axis=1).all():
            raise ValueError("Each patient must have at least one available expert")
        for j, m in enumerate(MODALITIES):
            if m not in self.x or self.x[m].shape != (n, len(self.features[m])):
                raise ValueError(f"Inconsistent {m} matrix/feature names")
            if len(set(self.features[m])) != len(self.features[m]):
                raise ValueError(f"Duplicate {m} features")
            if np.isinf(self.x[m]).any():
                raise ValueError(f"Infinite {m} values; represent missing values as NaN")
            if m == "mutation":
                a = self.x[m][np.isfinite(self.x[m])]
                if not np.isin(a, [0, 1]).all():
                    raise ValueError("Mutation features must be binary")
        if outcomes or self.time is not None:
            if self.time is None or self.event is None or self.time.shape != (n,) or self.event.shape != (n,):
                raise ValueError("Aligned survival time and event arrays required")
            if not (np.isfinite(self.time) & (self.time > 0)).all() or not np.isin(self.event, [0, 1]).all():
                raise ValueError("OS requires finite positive times and binary events")
        return self


def load_cohort(directory, outcomes=False):
    root = Path(directory)
    meta = json.loads((root / "manifest.json").read_text())
    c = pd.read_csv(root / "clinical.csv", dtype={"patient_id": str})
    if c.patient_id.tolist() != meta["patient_ids"]:
        raise ValueError("clinical.csv patient order differs from manifest")
    matrices, names = {}, {}
    for m in MODALITIES:
        info = meta["matrices"][m]
        names[m] = json.loads((root / info["features"]).read_text())
        shape = (len(names[m]), len(c))
        matrices[m] = (np.memmap(root / info["file"], dtype="float32", mode="r", shape=shape).T
                       if shape[0] else np.empty((len(c), 0), dtype="float32"))
    a = pd.read_csv(root / "availability.csv", index_col=0).reindex(c.patient_id)
    if a.isna().any().any():
        raise ValueError("Availability is missing patients")
    return Cohort(c.patient_id.to_numpy(), c[["age", "sex", "stage"]], matrices, names,
                  a[[*MODALITIES, "clinical"]].to_numpy(bool),
                  c.os_time_months.to_numpy(float) if "os_time_months" in c else None,
                  c.os_event.to_numpy(float) if "os_event" in c else None).validate(outcomes)


def write_cohort(data, directory, provenance=None):
    data.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    c = data.clinical.copy()
    c.insert(0, "patient_id", data.ids)
    if data.time is not None:
        c["os_time_months"], c["os_event"] = data.time, data.event
    c.to_csv(root / "clinical.csv", index=False)
    pd.DataFrame(data.available.astype(int), index=data.ids,
                 columns=[*MODALITIES, "clinical"]).rename_axis("patient_id").to_csv(root / "availability.csv")
    matrices = {}
    for m in MODALITIES:
        np.asarray(data.x[m].T, dtype="float32").tofile(root / f"{m}.dat")
        (root / f"{m}.features.json").write_text(json.dumps(data.features[m]))
        matrices[m] = {"file": f"{m}.dat", "features": f"{m}.features.json"}
    (root / "manifest.json").write_text(json.dumps({"format_version": 1, "patient_ids": data.ids.tolist(),
        "matrices": matrices, "provenance": provenance or {}, "fitted_preprocessing": False}, indent=2))


def primary_columns(columns, patients):
    wanted = set(patients)
    selected = {}
    # Deterministic: prefer 01A, then lexical aliquot ID. Do not average samples.
    for s in sorted(columns, key=lambda s: (not str(s)[13:].startswith("01A"), str(s))):
        if str(s).startswith("TCGA-") and str(s)[13:15] == "01" and str(s)[:12] in wanted:
            selected.setdefault(str(s)[:12], s)
    return selected


def prepare_xena(clinical_csv, xena_dir, cancer, output, rna_scale="raw", chunksize=2000,
                 feature_limit=None, mutation_coverage=None):
    """Stream all raw features to feature-major float32 files, including CpGs.

    rna_scale declares the *source* scale: raw means log2(x+1) is applied;
    log2p1 means input is already transformed. No cohort-fitted selection here.
    feature_limit is for ingestion smoke checks only, recorded in the manifest.
    """
    if feature_limit is not None and feature_limit < 1:
        raise ValueError("feature_limit must be positive")
    if rna_scale not in {"raw", "log2p1"}:
        raise ValueError("RNA source scale must be explicitly raw or log2p1")
    root, source = Path(output), Path(xena_dir)
    c = pd.read_csv(clinical_csv, dtype={"patient_id": str})
    required = {"patient_id", "age", "sex", "stage"}
    if not required <= set(c) or c.patient_id.duplicated().any() or c.patient_id.isna().any():
        raise ValueError("Clinical CSV requires unique patient_id and age/sex/stage columns")
    ids = c.patient_id.tolist()
    root.mkdir(parents=True, exist_ok=False)
    c.to_csv(root / "clinical.csv", index=False)
    availability = np.zeros((len(c), 5), dtype=bool)
    availability[:, 4] = c[["age", "sex", "stage"]].notna().any(axis=1)
    matrices, sources, mappings = {}, {}, {}
    files = {"rna": "star_tpm", "methylation": "methylation450", "cnv": "gene-level_ascat3"}
    for j, (m, suffix) in enumerate(files.items()):
        p = source / f"TCGA-{cancer}.{suffix}.tsv.gz"
        header = pd.read_csv(p, sep="\t", nrows=0).columns.tolist()
        chosen = primary_columns(header[1:], ids)
        if not chosen:
            raise ValueError(f"No clinical patients have primary-tumor samples in {p}")
        col_to_pid = {v: k for k, v in chosen.items()}
        names = []
        with (root / f"{m}.dat").open("wb") as out:
            for chunk in pd.read_csv(p, sep="\t", usecols=[header[0], *chosen.values()], chunksize=chunksize):
                if feature_limit is not None:
                    chunk = chunk.iloc[:max(0, feature_limit - len(names))]
                if chunk.empty:
                    break
                feature_names = chunk[header[0]].astype(str).tolist()
                values = chunk.drop(columns=header[0]).rename(columns=col_to_pid).reindex(columns=ids)
                a = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
                if m == "rna" and rna_scale == "raw":
                    if np.nanmin(a) < 0:
                        raise ValueError("Negative raw RNA values; specify source scale correctly")
                    a = np.log2(a + 1.)
                if m == "methylation" and ((a[np.isfinite(a)] < 0).any() or (a[np.isfinite(a)] > 1).any()):
                    raise ValueError("Expected methylation beta values in [0,1]")
                availability[:, j] |= np.isfinite(a).any(axis=0)
                a.tofile(out)
                names.extend(feature_names)
                if feature_limit is not None and len(names) >= feature_limit:
                    break
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate feature IDs in {p}; resolve explicitly before training")
        (root / f"{m}.features.json").write_text(json.dumps(names))
        matrices[m] = {"file": f"{m}.dat", "features": f"{m}.features.json"}
        sources[m] = {"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        mappings[m] = chosen
        print(f"Prepared {m}: {len(names)} features, {availability[:, j].sum()} patients", flush=True)
    p = source / f"TCGA-{cancer}.somaticmutation_wxs.tsv.gz"
    mut = pd.read_csv(p, sep="\t", usecols=["sample", "gene"])
    mut = mut[mut["sample"].astype(str).str[13:15].eq("01")].dropna(subset=["sample"])
    mut["patient_id"] = mut["sample"].str[:12]
    mut = mut[mut.patient_id.isin(ids)]
    genes = sorted(mut.gene.dropna().astype(str).unique())
    if feature_limit is not None:
        genes = genes[:feature_limit]
    covered = set(mut.patient_id)
    if mutation_coverage is not None:
        # Assayed patients with no called variants must be supplied explicitly.
        cov = pd.read_csv(mutation_coverage)
        covered = set(cov.loc[cov.available.eq(1), "patient_id"].astype(str))
    availability[:, 3] = [p in covered for p in ids]
    a = np.full((len(genes), len(ids)), np.nan, dtype="float32")
    a[:, availability[:, 3]] = 0
    gi, pi = {g: i for i, g in enumerate(genes)}, {p: i for i, p in enumerate(ids)}
    for pid, gene in mut[["patient_id", "gene"]].itertuples(index=False, name=None):
        if gene in gi and pid in covered:
            a[gi[gene], pi[pid]] = 1
    a.tofile(root / "mutation.dat")
    (root / "mutation.features.json").write_text(json.dumps(genes))
    matrices["mutation"] = {"file": "mutation.dat", "features": "mutation.features.json"}
    sources["mutation"] = {"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    pd.DataFrame(availability.astype(int), index=ids, columns=[*MODALITIES, "clinical"]).rename_axis("patient_id").to_csv(root / "availability.csv")
    manifest = {"format_version": 1, "patient_ids": ids, "matrices": matrices, "fitted_preprocessing": False,
                "source_rna_scale": rna_scale, "prepared_rna_scale": "log2p1", "feature_limit_smoke_only": feature_limit,
                "sources": sources, "sample_mappings": mappings, "mutation_coverage_policy": "explicit_assay_manifest" if mutation_coverage else "presence_of_variant_records",
                "clinical_source": str(Path(clinical_csv).resolve()),
                "clinical_sha256": hashlib.sha256(Path(clinical_csv).read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return load_cohort(root, outcomes="os_time_months" in c)


def synthetic_cohort(n=100, p=12, seed=0):
    """Synthetic fixtures for software checks, never manuscript result data."""
    rng = np.random.default_rng(seed)
    x = {m: rng.normal(size=(n, p)).astype("float32") for m in MODALITIES}
    x["methylation"] = rng.uniform(size=(n, p)).astype("float32")
    x["mutation"] = rng.binomial(1, .3, size=(n, p)).astype("float32")
    available = np.ones((n, 5), bool)
    for j, m in enumerate(MODALITIES):
        absent = rng.random(n) < .05
        available[absent, j] = False
        x[m][absent] = np.nan
    age = rng.uniform(40, 85, n)
    risk = np.exp(.4 * np.nan_to_num(x["rna"][:, 0]) + .02 * (age - 60))
    death = rng.exponential(65 / risk)
    censor = rng.uniform(30, 110, n)
    c = pd.DataFrame({"age": age, "sex": rng.choice(["male", "female"], n),
                      "stage": rng.choice(["Stage II", "Stage III", "Stage IV", None], n)})
    return Cohort(np.array([f"synthetic-{i:04d}" for i in range(n)]), c, x,
                  {m: [f"{m}_{i}" for i in range(p)] for m in MODALITIES}, available,
                  np.maximum(.01, np.minimum(death, censor)), (death <= censor).astype(int)).validate(True)


def import_tables(clinical_csv, matrices, output, availability_csv=None):
    """Import patient-row feature-column CSVs with stable feature identifiers."""
    c = pd.read_csv(clinical_csv, dtype={"patient_id": str})
    ids = c.patient_id.to_numpy()
    x, features = {}, {}
    available = np.zeros((len(c), 5), bool)
    for j, m in enumerate(MODALITIES):
        if m not in matrices:
            x[m], features[m] = np.empty((len(c), 0)), []
            continue
        table = pd.read_csv(matrices[m], index_col="patient_id")
        table.index = table.index.astype(str)
        if not table.index.is_unique or not table.columns.is_unique:
            raise ValueError(f"{m} CSV must have unique patient/feature identifiers")
        aligned = table.reindex(ids).apply(pd.to_numeric, errors="raise")
        x[m], features[m] = aligned.to_numpy("float32"), aligned.columns.tolist()
        available[:, j] = np.isfinite(x[m]).any(axis=1)
    available[:, 4] = c[["age", "sex", "stage"]].notna().any(axis=1)
    if availability_csv is not None:
        a = pd.read_csv(availability_csv, index_col="patient_id").reindex(ids)[[*MODALITIES, "clinical"]]
        if a.isna().any().any() or not np.isin(a.to_numpy(), [0, 1]).all():
            raise ValueError("Explicit availability must be complete and binary")
        available = a.to_numpy(bool)
    data = Cohort(ids, c[["age", "sex", "stage"]], x, features, available,
                  c.os_time_months.to_numpy(float) if "os_time_months" in c else None,
                  c.os_event.to_numpy(float) if "os_event" in c else None).validate()
    write_cohort(data, output, {"clinical_csv": str(Path(clinical_csv).resolve()),
                               "matrix_paths": {m: str(Path(p).resolve()) for m, p in matrices.items()},
                               "input_scale": "User-supplied canonical scale; no log transform applied"})
    return data


def extract_xena_clinical(xena_dir, cancer, output):
    """Normalize clinical fields and OS; emit excluded records separately."""
    source = Path(xena_dir)
    clinical = pd.read_csv(source/f"TCGA-{cancer}.clinical.tsv.gz", sep="\t", low_memory=False)
    survival = pd.read_csv(source/f"TCGA-{cancer}.survival.tsv.gz", sep="\t")
    clinical["patient_id"] = clinical["sample"].astype(str).str[:12]
    # Prefer primary tumor clinical record, then lexical sample barcode.
    clinical["primary"] = clinical["sample"].astype(str).str[13:15].eq("01")
    clinical = clinical.sort_values(["primary", "sample"], ascending=[False, True]).drop_duplicates("patient_id").set_index("patient_id")
    survival["patient_id"] = survival["_PATIENT"].astype(str)
    conflicts = survival.groupby("patient_id")[["OS.time", "OS"]].nunique(dropna=False).max(axis=1) > 1
    if conflicts.any():
        raise ValueError("Conflicting patient-level survival records require adjudication")
    survival = survival.drop_duplicates("patient_id").set_index("patient_id")
    c = pd.DataFrame(index=survival.index)
    c["os_time_months"] = pd.to_numeric(survival["OS.time"], errors="coerce")/30.4375
    c["os_event"] = pd.to_numeric(survival["OS"], errors="coerce")
    if "age_at_index.demographic" in clinical:
        c["age"] = pd.to_numeric(clinical["age_at_index.demographic"], errors="coerce")
    elif "age_at_diagnosis.diagnoses" in clinical:
        c["age"] = pd.to_numeric(clinical["age_at_diagnosis.diagnoses"], errors="coerce")/365.25
    else:
        c["age"] = np.nan
    c["sex"] = clinical.get("gender.demographic", pd.Series(dtype=str))
    c["stage"] = clinical.get("ajcc_pathologic_stage.diagnoses", pd.Series(dtype=str))
    valid = np.isfinite(c.os_time_months) & (c.os_time_months > 0) & c.os_event.isin([0, 1])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".excluded.csv").exists():
        raise FileExistsError(output)
    c[~valid].assign(exclusion_reason="invalid_or_missing_OS").to_csv(output.with_suffix(".excluded.csv"))
    c[valid].to_csv(output)
    return {"included": int(valid.sum()), "excluded": int((~valid).sum()), "days_per_month": 30.4375}
