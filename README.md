# CLPsych Teacher-Labeling Pipeline

A data-labeling pipeline that turns raw CLPsych shared-task Reddit timelines into
validated, structured auxiliary-label predictions for student-model training.
Follows the teacher-auxiliary-label plan: split → lock test → generate →
quality-gate → majority-vote → route human audit → build student data.

## ⚠️ Ethics first
This processes sensitive mental-health / crisis text.
- Teacher generation calls the Gemini API, which sends de-identified post text off-machine.
  Confirm your institution's policy clearly allows this BEFORE running — there is no
  local/offline teacher path in this pipeline.
- Teacher labels are NOISY training supervision, never final ground truth.
- Final results must come from the LOCKED, human-relabeled test set.
- `LOCKED_test.csv` (held-out timelines) must never pass through teacher generation.

## Stages

```
CLPsych timeline JSONs ──prepare_clpsych──▶ input_posts.csv
                       ──create_data_splits──▶ splits/ (+ LOCKED_test.csv)
                       ──generate_aux_labels──▶ raw_primary_runs.jsonl   (train/dev only)
                       ──quality_gates──▶ majority_candidates.jsonl + rejected.jsonl + metrics
                       ──build_student_sft──▶ student_sft.jsonl
```

## What is being labeled
Each **post** in a user timeline gets predicted values for:

| Field | Values | Meaning |
|---|---|---|
| `Switch` | `0` or `S` | Crisis state switch occurred at this post |
| `Escalation` | `0` or `E` | Escalation in risk at this post |
| `Well-being` | `1-10` or `null` | Subjective well-being score (null if not inferable) |
| `adaptive_presence` | `1-5` or `null` | Strength of adaptive coping signals |
| `maladaptive_presence` | `1-5` or `null` | Strength of maladaptive signals |

See `config/clpsych_schema.md` for the full rubric.

## Quick start

```bash
# 0. extract the CLPsych shared-task archive (password-protected DUA data --
#    you'll be prompted for the password; don't share it in chat/commit it)
unzip train_tasks12_clpsych2026.zip -d raw/

# 1. flatten every timeline's posts + de-identify
python src/prepare_clpsych.py \
    --input-dir raw/train_tasks12 \
    --output data/processed/input_posts.csv

# 2. timeline-level stratified split; locks the test set (never split a
#    timeline's posts across train/dev/test)
python src/create_data_splits.py \
    --input data/processed/input_posts.csv \
    --output-dir data/processed/splits --seed 42

# 2b. (optional but recommended) write a test-excluded copy before generation,
#     so held-out timeline text never sits on disk anywhere near the teacher call
#     -- belt-and-suspenders on top of --allowed-splits below.
python -c "
import pandas as pd
df = pd.read_csv('data/processed/splits/all_with_splits.csv')
df[df.split.isin(['train','dev'])].to_csv('data/processed/splits/train_dev_for_teacher.csv', index=False)
"

# 3. PILOT: generate 200 posts first, measure, freeze prompt, THEN scale
#    (requires GEMINI_API_KEY, or GOOGLE_API_KEY, in the environment)
python src/generate_aux_labels.py \
    --input data/processed/splits/train_dev_for_teacher.csv \
    --output data/synthetic_aux/raw_primary_runs.jsonl \
    --model gemini-2.5-flash --runs 3 --limit 200

# 4. validate + majority-vote/mean-aggregate + route human audit
python src/quality_gates.py \
    --input data/synthetic_aux/raw_primary_runs.jsonl \
    --accepted data/synthetic_aux/majority_candidates.jsonl \
    --rejected data/synthetic_aux/rejected.jsonl \
    --report data/synthetic_aux/pilot_metrics.json

# 5. build student training data
python src/build_student_sft.py \
    --input data/synthetic_aux/majority_candidates.jsonl \
    --output data/synthetic_aux/student_sft.jsonl --mode sft
```

## The pilot gate (do this before the full run)
After stage 4 on 200 posts, look at `pilot_metrics.json`:
- `valid_run_rate` — if low, tighten the prompt (malformed JSON / out-of-range values).
- `switch_distribution` — sanity-check it isn't collapsing to all-`0`.
- `rejection_reasons` — tells you exactly what to fix in the prompt.
Freeze the improved prompt as `teacher_prompt_v2.txt`, then run the full set.

## Compute
- Teacher: Gemini API — no local GPU needed, runs from any machine with network access.
- Student QLoRA (≤3B): comfortable on T4/P100.

## Files
```
config/   schema.json, clpsych_schema.md, teacher_prompt_v1.txt
src/      prepare_clpsych.py, create_data_splits.py, generate_aux_labels.py,
          quality_gates.py, build_student_sft.py
notebooks/ colab_clpsych_aux_labels.ipynb -- run generation on Colab (no GPU needed)
```

## Not yet included (next layers from the plan)
- disagreement teacher (second model family)
- separate rubric judge model
- human-audit review UI
These add robustness; the current pipeline is the single-teacher MVP with
majority vote + quality gates + audit routing.
