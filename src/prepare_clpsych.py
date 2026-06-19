"""
prepare_clpsych.py
Load CLPsych shared-task timeline JSON files, flatten every post across every
timeline, de-identify, and emit a unified input_posts.csv.

Each input file is expected to look like:
    { "timeline_id": "...", "posts": [
        { "post_id": ..., "post_index": ..., "post": "...",
          "Switch": "0"|"S", "Escalation": "0"|"E", "Well-being": 1-10|null }, ...
    ] }

Output columns: id,user_id,source,text,gold_label,post_id,post_index,
                 gold_switch,gold_escalation,gold_wellbeing

`user_id` is the timeline_id (posts from one timeline must never be split
across train/dev/test -- create_data_splits.py groups by user_id already).
`gold_label` is a coarse per-post category (switch/escalation/stable) used
only for stratifying the split, not as a training target.

Usage:
    python prepare_clpsych.py --input-dir raw/train_tasks12 \
                              --output data/processed/input_posts.csv
"""
import argparse, glob, html, json, os, re
import pandas as pd

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
USER_RE    = re.compile(r"/?u/[A-Za-z0-9_\-]+")
SUBR_RE    = re.compile(r"/?r/[A-Za-z0-9_\-]+")
EMAIL_RE   = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
HANDLE_RE  = re.compile(r"@[A-Za-z0-9_]+")


def deidentify(text):
    text = html.unescape(text)
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = USER_RE.sub("[USER]", text)
    text = SUBR_RE.sub("[SUBREDDIT]", text)
    text = HANDLE_RE.sub("[HANDLE]", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def coarse_label(switch, escalation):
    if switch == "S":
        return "switch"
    if escalation == "E":
        return "escalation"
    return "stable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="directory of CLPsych timeline JSON files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-chars", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not paths:
        raise ValueError(f"No .json timeline files found in {args.input_dir}")

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            timeline = json.load(f)
        tid = timeline["timeline_id"]
        for post in timeline.get("posts", []):
            text = deidentify(str(post.get("post", "")))
            if len(text) < args.min_chars:
                continue
            if len(text) > args.max_chars:
                text = text[: args.max_chars]
            switch = post.get("Switch")
            escalation = post.get("Escalation")
            rows.append({
                "id": f"{tid}_{post.get('post_id')}",
                "user_id": tid,
                "source": "clpsych",
                "text": text,
                "gold_label": coarse_label(switch, escalation),
                "post_id": post.get("post_id"),
                "post_index": post.get("post_index"),
                "gold_switch": switch,
                "gold_escalation": escalation,
                "gold_wellbeing": post.get("Well-being"),
            })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} posts from {len(paths)} timelines -> {args.output}")
    print("Coarse label distribution (for split stratification only):")
    print(out.gold_label.value_counts())


if __name__ == "__main__":
    main()
