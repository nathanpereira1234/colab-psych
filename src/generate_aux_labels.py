"""
generate_aux_labels.py
Generate CLPsych auxiliary teacher labels (Switch / Escalation / Well-being /
adaptive_presence / maladaptive_presence) with the Gemini API as the teacher model.
Uses native generate_content() with structured outputs (response_json_schema)
so every response is guaranteed valid JSON matching the schema.

Runs N stochastic generations per post (default 3). Only processes the splits in
--allowed-splits (default train,dev) so the locked test set is never touched.
Checkpoints to disk so a disconnect doesn't lose work.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.

Example:
    python generate_aux_labels.py --input splits/all_with_splits.csv \
        --output raw_primary_runs.jsonl --model gemini-2.5-flash \
        --runs 3 --limit 200
"""
import argparse, json, os
import pandas as pd
from tqdm.auto import tqdm

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "teacher_prompt_v1.txt")


def load_prompt_template():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def extract_json(s):
    s = s.replace("```json", "").replace("```", "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1:
        s = s[a:b + 1]
    return json.loads(s)


# Mirrors the schema described in config/teacher_prompt_v1.txt, as a real JSON
# Schema for response_json_schema (structured outputs) instead of prose + regex extraction.
# Numeric min/max aren't in Gemini's supported JSON Schema subset, so the 1-10 / 1-5
# ranges are enforced by quality_gates.py instead of by the schema itself.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Switch": {"type": "string", "enum": ["0", "S"]},
        "Escalation": {"type": "string", "enum": ["0", "E"]},
        "Well-being": {"type": ["integer", "null"]},
        "adaptive_presence": {"type": ["integer", "null"]},
        "maladaptive_presence": {"type": ["integer", "null"]},
    },
    "required": ["Switch", "Escalation", "Well-being", "adaptive_presence", "maladaptive_presence"],
    "additionalProperties": False,
}


class GeminiBackend:
    # Google explicitly recommends leaving sampling params at their defaults on
    # Gemini 3.x models -- if you switch --model to a gemini-3* id, drop --temperature
    # (don't pass it) to follow that guidance; 2.x models support it normally.
    def __init__(self, model_id, temperature=0.4):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment
        self.model_id = model_id
        self.temperature = temperature

    def generate(self, prompt):
        # The static schema/rubric instructions (everything before POST:) are identical
        # on every call -- send them as the system instruction, separate from the
        # per-post user content.
        system_text, _, rest = prompt.partition("POST:")
        config = self.types.GenerateContentConfig(
            system_instruction=system_text.strip(),
            temperature=self.temperature,
            max_output_tokens=256,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
        )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents="POST:" + rest,
            config=config,
        )
        return response.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--allowed-splits", default="train,dev")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    allowed = set(args.allowed_splits.split(","))
    if args.split_column in df.columns:
        df = df[df[args.split_column].isin(allowed)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Generating for {len(df)} posts x {args.runs} runs "
          f"(splits={allowed}) -- locked test excluded")

    backend = GeminiBackend(args.model, temperature=args.temperature)

    template = load_prompt_template()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
            rd = row._asdict()
            post = str(rd[args.text_column])
            prompt = template.replace("{post_text}", post)
            for run in range(args.runs):
                raw = None
                parsed = None
                try:
                    raw = backend.generate(prompt)
                    parsed = extract_json(raw)
                except Exception as e:
                    parsed = None
                rec = {
                    "id": rd[args.id_column],
                    "run": run,
                    "text": post,
                    "gold_switch": rd.get("gold_switch"),
                    "gold_escalation": rd.get("gold_escalation"),
                    "gold_wellbeing": rd.get("gold_wellbeing"),
                    "parsed": parsed,           # None if JSON failed (quality gate will drop)
                    "raw": raw if parsed is None else None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            if (i + 1) % args.checkpoint_every == 0:
                fout.flush()
    print(f"Wrote {n_written} raw runs -> {args.output}")


if __name__ == "__main__":
    main()
