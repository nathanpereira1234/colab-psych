"""
build_student_sft.py
Convert accepted auxiliary labels into student training records. Two modes:
  --mode sft : target is the full JSON object (plain SFT)
  --mode aux : emit per-field components so train_auxiliary.py can apply
               separate masked losses (switch/escalation/wellbeing/...)

Usage:
    python build_student_sft.py --input accepted.jsonl --output student_sft.jsonl --mode sft
"""
import argparse, json

INSTR = ("Read this single Reddit post from a mental-health support forum and predict: "
         "Switch (\"S\"=crisis-state switch, \"0\"=no switch), "
         "Escalation (\"E\"=escalation in risk/distress, \"0\"=none), "
         "Well-being (1-10 or null), adaptive_presence (1-5 or null), "
         "maladaptive_presence (1-5 or null). Return JSON.")


def to_student_json(rec):
    keep = ["Switch", "Escalation", "Well-being", "adaptive_presence", "maladaptive_presence"]
    return {k: rec[k] for k in keep if k in rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["sft", "aux"], default="sft")
    ap.add_argument("--exclude-human-routed", action="store_true",
                    help="drop rows flagged for human audit (use only auto-accepted)")
    args = ap.parse_args()

    n = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            if args.exclude_human_routed and rec.get("_route_human"):
                continue
            text = rec.get("_text", "")
            target = to_student_json(rec)
            prompt = f"{INSTR}\n\nPOST:\n{text}\n\nJSON:"
            if args.mode == "sft":
                out = {"prompt": prompt,
                       "completion": " " + json.dumps(target, ensure_ascii=False)}
            else:  # aux: keep fields separate for masked multi-loss training
                out = {"prompt": prompt,
                       "fields": target,
                       "id": rec.get("_id")}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} student records ({args.mode}) -> {args.output}")


if __name__ == "__main__":
    main()
