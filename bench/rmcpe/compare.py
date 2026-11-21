"""Pool paired counts and bootstrap the held-out comparison by singer."""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("rmvpe", "fcpe", "rmcpe")
KEYS = ("frames", "ref_voiced", "ref_unvoiced", "correct_50", "missed_voice", "false_voice", "octave_errors")
METRICS = ("overall_accuracy_pct", "voiced_within_50_pct", "missed_voice_pct", "false_voice_pct", "octave_pct")


def rates(counts):
    n, voiced, unvoiced, correct, miss, false, octave = np.moveaxis(counts, -1, 0)
    return 100 * np.stack([(correct + unvoiced - false) / n, correct / voiced, miss / voiced,
                           false / unvoiced, octave / voiced], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    runs = {m: json.loads((args.work / f"{m}-all.json").read_text()) for m in METHODS}
    records = {}
    for track, row in runs["rmvpe"]["tracks"].items():
        counts = {}
        for method, run in runs.items():
            r = run["tracks"][track]
            if (r["singer"], r["split"]) != (row["singer"], row["split"]):
                raise ValueError(f"Different evaluation partitions for {track}")
            if any(r["counts"][key] != row["counts"][key] for key in ("frames", "ref_voiced", "ref_unvoiced")):
                raise ValueError(f"Different annotated frames for {track}")
            counts[method] = [r["counts"][key] for key in KEYS]
        records[track] = {"singer": row["singer"], "split": row["split"], "counts": counts}
    aggregates = {}
    for split in ("calibration", "heldout", "all"):
        tracks = [r for r in records.values() if split == "all" or r["split"] == split]
        singers = sorted({r["singer"] for r in tracks})
        matrix = np.stack([np.sum([[r["counts"][m] for m in METHODS] for r in tracks if r["singer"] == s],
                                  axis=0) for s in singers])
        totals = matrix.sum(axis=0)
        micro = rates(totals)
        macro = rates(matrix).mean(axis=0)
        aggregate = {"singers": len(singers), "tracks": len(tracks), "methods": {}}
        for i, method in enumerate(METHODS):
            aggregate["methods"][method] = {"counts": totals[i].tolist(),
                "micro": dict(zip(METRICS, micro[i].round(3).tolist())),
                "singer_macro": dict(zip(METRICS, macro[i].round(3).tolist()))}
        if split == "heldout":
            rng = np.random.default_rng(1027)
            indices = rng.integers(len(singers), size=(5000, len(singers)))
            bootstrap = rates(matrix[indices].sum(axis=1))
            aggregate["rmcpe_minus_baseline_pp"] = {}
            for i in (0, 1):
                intervals = np.percentile(bootstrap[:, 2] - bootstrap[:, i], [2.5, 97.5], axis=0)
                aggregate["rmcpe_minus_baseline_pp"][METHODS[i]] = {
                    k: [round(float(micro[2,j] - micro[i,j]),3), *intervals[:,j].round(3).tolist()]
                    for j,k in enumerate(METRICS)}
        aggregates[split] = aggregate
    output = {"dataset": "Vocadito v3", "archive_md5": runs["rmvpe"]["archive_md5"],
              "device": runs["rmvpe"]["device"], "count_columns": KEYS,
              "extraction_wall_s": {m:r["fresh_or_cached_extraction_s"] for m,r in runs.items()},
              "bootstrap": {"seed": 1027, "replicates": 5000, "unit": "held-out singer",
                            "interval": "paired point estimate and 2.5/97.5 percentiles, percentage points"},
              "aggregates": aggregates, "tracks": records}
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(aggregates["heldout"], indent=2))


if __name__ == "__main__":
    main()
