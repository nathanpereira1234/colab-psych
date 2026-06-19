"""
create_data_splits.py
Split into train/dev/test by USER (not post) to prevent leakage, stratified by
gold_label where possible. The test split is LOCKED: it is written separately and
must NOT be sent through the teacher-generation pipeline.

Usage:
    python create_data_splits.py --input data/processed/input_posts.csv \
                                 --output-dir data/processed/splits --seed 42
"""
import argparse, os
import pandas as pd
from collections import defaultdict
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--dev", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train + args.dev + args.test - 1.0) < 1e-6, "splits must sum to 1"
    random.seed(args.seed)

    df = pd.read_csv(args.input)

    # group by user, take each user's majority label for stratification
    user_label = {}
    for uid, g in df.groupby("user_id"):
        user_label[uid] = g["gold_label"].mode().iloc[0]

    # bucket users by label, shuffle within bucket, then split each bucket
    buckets = defaultdict(list)
    for uid, lab in user_label.items():
        buckets[lab].append(uid)

    split_of = {}
    for lab, users in buckets.items():
        random.shuffle(users)
        n = len(users)
        n_tr = int(n * args.train)
        n_dev = int(n * args.dev)
        for u in users[:n_tr]:            split_of[u] = "train"
        for u in users[n_tr:n_tr+n_dev]:  split_of[u] = "dev"
        for u in users[n_tr+n_dev:]:      split_of[u] = "test"

    df["split"] = df["user_id"].map(split_of)

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "all_with_splits.csv"), index=False)

    # write test separately and lock it
    test_df = df[df.split == "test"].copy()
    test_df.to_csv(os.path.join(args.output_dir, "LOCKED_test.csv"), index=False)

    print("Split sizes (posts):")
    print(df.split.value_counts())
    print("\nUnique users per split:")
    print(df.groupby("split")["user_id"].nunique())
    print(f"\nLOCKED test written -> {args.output_dir}/LOCKED_test.csv")
    print("Do NOT send LOCKED_test through teacher generation. Relabel it by hand.")


if __name__ == "__main__":
    main()
