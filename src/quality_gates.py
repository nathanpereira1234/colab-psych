"""
quality_gates.py
Validate raw teacher runs against the CLPsych schema, then aggregate the N runs
per post: majority vote for Switch/Escalation, mean for the 1-10/1-5 scales.
Writes accepted candidates + a rejection log + a pilot metrics summary.

Usage:
    python quality_gates.py --input raw_primary_runs.jsonl \
        --accepted majority_candidates.jsonl --rejected rejected.jsonl \
        --report pilot_metrics.json
"""
import argparse, json
from collections import defaultdict, Counter

REQUIRED = ["Switch", "Escalation", "Well-being", "adaptive_presence", "maladaptive_presence"]


def validate(rec):
    """Return (ok, reasons[]) for a single parsed run."""
    reasons = []
    p = rec.get("parsed")
    if p is None:
        return False, ["malformed_or_unparsed_json"]
    for f in REQUIRED:
        if f not in p:
            reasons.append(f"missing_field:{f}")
            continue
        v = p[f]
        if f == "Switch" and v not in ("0", "S"):
            reasons.append("switch_out_of_range")
        elif f == "Escalation" and v not in ("0", "E"):
            reasons.append("escalation_out_of_range")
        elif f == "Well-being" and v is not None and (not isinstance(v, int) or not (1 <= v <= 10)):
            reasons.append("wellbeing_out_of_range")
        elif f in ("adaptive_presence", "maladaptive_presence") and v is not None \
                and (not isinstance(v, int) or not (1 <= v <= 5)):
            reasons.append(f"{f}_out_of_range")
    return (len(reasons) == 0), reasons


def majority(values):
    """Majority vote over non-null values. Returns (value, agreement_fraction)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, 0.0
    val, n = Counter(vals).most_common(1)[0]
    return val, round(n / len(vals), 3)


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--rejected", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    by_id = defaultdict(list)
    total_runs = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["id"]].append(rec)
            total_runs += 1

    accepted, rejected = [], []
    reason_counter = Counter()
    switch_counter = Counter()
    n_valid_runs = 0

    for pid, runs in by_id.items():
        valid = []
        for r in runs:
            ok, reasons = validate(r)
            if ok:
                valid.append(r)
                n_valid_runs += 1
            else:
                reason_counter.update(reasons)
                rejected.append({"id": pid, "run": r["run"], "reasons": reasons})

        if not valid:
            continue

        switch, switch_agreement = majority(v["parsed"]["Switch"] for v in valid)
        escalation, escalation_agreement = majority(v["parsed"]["Escalation"] for v in valid)
        wellbeing = mean_or_none(v["parsed"]["Well-being"] for v in valid)
        adaptive = mean_or_none(v["parsed"]["adaptive_presence"] for v in valid)
        maladaptive = mean_or_none(v["parsed"]["maladaptive_presence"] for v in valid)

        out = {
            "_id": pid,
            "_text": valid[0].get("text"),
            "_gold_switch": valid[0].get("gold_switch"),
            "_gold_escalation": valid[0].get("gold_escalation"),
            "_gold_wellbeing": valid[0].get("gold_wellbeing"),
            "Switch": switch,
            "Escalation": escalation,
            "Well-being": wellbeing,
            "adaptive_presence": adaptive,
            "maladaptive_presence": maladaptive,
            "_switch_agreement": switch_agreement,
            "_escalation_agreement": escalation_agreement,
            "_n_valid_runs": len(valid),
        }
        # human-audit routing: a detected switch, or any run-to-run disagreement
        out["_route_human"] = (
            switch == "S"
            or switch_agreement < 1.0
            or escalation_agreement < 1.0
        )
        accepted.append(out)
        switch_counter[switch] += 1

    with open(args.accepted, "w", encoding="utf-8") as f:
        for a in accepted:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(args.rejected, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "posts_total": len(by_id),
        "runs_total": total_runs,
        "valid_runs": n_valid_runs,
        "valid_run_rate": round(n_valid_runs / total_runs, 3) if total_runs else 0,
        "posts_accepted": len(accepted),
        "switch_distribution": dict(switch_counter),
        "rejection_reasons": dict(reason_counter),
        "routed_to_human": sum(1 for a in accepted if a["_route_human"]),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
