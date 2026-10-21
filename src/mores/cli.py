"""Command-line interfaces; paths and runtime choices are never hard-coded."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config
from .data import (load_cohort, prepare_xena, synthetic_cohort, write_cohort,
                   import_tables, extract_xena_clinical)


def save_prediction(prediction, ids, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root/"predictions.npz", patient_ids=np.asarray(ids, dtype=str), **prediction)
    diagnostics = {k: v for k, v in prediction.items() if np.asarray(v).ndim == 1 and k != "times"}
    pd.DataFrame({"patient_id": ids, **diagnostics}).to_csv(root/"diagnostics.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description="MoReS multi-omics survival modeling")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-xena", help="Stream primary-tumor matrices into a raw data cache")
    prep.add_argument("--clinical", required=True)
    prep.add_argument("--xena-dir", required=True)
    prep.add_argument("--cancer", default="BLCA")
    prep.add_argument("--rna-source-scale", choices=["raw", "log2p1"], required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--feature-limit", type=int, help="Smoke checks only; recorded in manifest")
    prep.add_argument("--mutation-coverage", help="Optional patient_id,available assay coverage CSV")
    clinical = sub.add_parser("extract-clinical")
    clinical.add_argument("--xena-dir", required=True)
    clinical.add_argument("--cancer", required=True)
    clinical.add_argument("--output", required=True)
    imp = sub.add_parser("import-tables", help="Import patient-row CSVs, already on canonical scales")
    imp.add_argument("--clinical", required=True)
    for m in ["rna", "methylation", "cnv", "mutation"]:
        imp.add_argument(f"--{m}")
    imp.add_argument("--availability")
    imp.add_argument("--output", required=True)
    synth = sub.add_parser("make-demo", help="Synthetic data for software validation only")
    synth.add_argument("--patients", type=int, default=256)
    synth.add_argument("--features", type=int, default=8)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--output", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--data", required=True)
    for name in ["train", "cv"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--output", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--model", required=True)
    pred.add_argument("--data", required=True)
    pred.add_argument("--times", nargs="+", type=float)
    pred.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate", help="Evaluate a frozen model on a disjoint cohort")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--horizon", type=float, default=36.)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--perturbation", choices=["rna_mask", "methylation_noise", "cnv_rescale", "mutation_missing", "modality_drop", "platform_shift"])
    evaluation.add_argument("--severity", type=float, default=.15)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare-xena":
        result = prepare_xena(args.clinical, args.xena_dir, args.cancer, args.output,
                              args.rna_source_scale, feature_limit=args.feature_limit,
                              mutation_coverage=args.mutation_coverage)
        print(f"Prepared {len(result)} patients")
    elif args.command == "extract-clinical":
        print(json.dumps(extract_xena_clinical(args.xena_dir, args.cancer, args.output)))
    elif args.command == "import-tables":
        matrices = {m: getattr(args, m) for m in ["rna", "methylation", "cnv", "mutation"] if getattr(args, m)}
        import_tables(args.clinical, matrices, args.output, args.availability)
    elif args.command == "make-demo":
        write_cohort(synthetic_cohort(args.patients, args.features, args.seed), args.output,
                     {"synthetic": True, "purpose": "software validation only", "seed": args.seed})
    elif args.command == "audit":
        d = load_cohort(args.data)
        info = {"n_patients": len(d), "features": {m: len(v) for m, v in d.features.items()},
                "available": dict(zip([*d.x, "clinical"], d.available.sum(0).astype(int).tolist())),
                "joint_eligible": int((d.available[:, :4].sum(1) >= 3).sum())}
        if d.time is not None:
            info.update(events=int(d.event.sum()), median_time=float(np.median(d.time)))
        print(json.dumps(info, indent=2))
    elif args.command in {"train", "cv"}:
        from .pipeline import fit, cross_validate, runtime_info
        cfg = Config.load(args.config)
        d = load_cohort(args.data, outcomes=True)
        meta = json.loads((Path(args.data)/"manifest.json").read_text())
        prepared_scale = meta.get("prepared_rna_scale")
        if prepared_scale and prepared_scale != cfg.rna_scale:
            raise ValueError("Config RNA scale differs from prepared input scale")
        if args.command == "cv":
            cross_validate(d, cfg, args.output)
        else:
            root = Path(args.output)
            root.mkdir(parents=True, exist_ok=False)
            cfg.save(root/"config.json")
            model = fit(d, cfg)
            model.save(root/"model.joblib")
            (root/"provenance.json").write_text(json.dumps(model.provenance, indent=2))
            (root/"runtime.json").write_text(json.dumps(runtime_info(), indent=2))
        (Path(args.output)/"data_manifest.json").write_text(json.dumps(meta, indent=2))
    elif args.command == "predict":
        from .pipeline import MoReS
        model, d = MoReS.load(args.model), load_cohort(args.data)
        save_prediction(model.predict(d, args.times), d.ids, args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate_frozen
        from .pipeline import MoReS
        model, data = MoReS.load(args.model), load_cohort(args.data, outcomes=True)
        evaluate_frozen(model, data, args.output, args.horizon, args.perturbation, args.severity)


if __name__ == "__main__":
    main()
