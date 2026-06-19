# CLPsych Auxiliary-Label Schema

The teacher annotates ONE Reddit post at a time (no surrounding timeline context)
and returns five fields.

## Switch — crisis state switch
- `"S"` — a switch into a notably different (typically worse) mental state is evident
  in this post relative to the author's baseline.
- `"0"` — no switch detected.

## Escalation — risk/distress escalation
- `"E"` — escalation is evident (increased hopelessness, self-harm ideation, urgency).
- `"0"` — no escalation detected.

## Well-being — subjective well-being
Integer 1 (very low, severe distress) to 10 (very high, flourishing), or `null` if the
post doesn't contain enough emotional content to rate.

## adaptive_presence — adaptive coping signal strength
Integer 1 (very weak/absent) to 5 (very strong), or `null` if not inferable. Adaptive
signals: help-seeking, connection with others, self-efficacy, positive reappraisal.

## maladaptive_presence — maladaptive signal strength
Integer 1 (very weak/absent) to 5 (very strong), or `null` if not inferable.
Maladaptive signals: hopelessness, isolation, self-harm, suicidal ideation, cognitive
distortions.

## Notes
- `Switch` and `Escalation` are the two gold-labeled fields in the CLPsych 2026 Tasks
  1 & 2 training data (`gold_switch`, `gold_escalation` after `prepare_clpsych.py`).
  `Well-being` also has a gold value in the source data; `adaptive_presence` and
  `maladaptive_presence` have no gold reference — they're auxiliary signals only,
  evaluated for inter-run consistency rather than gold agreement.
- Prefer `null` over guessing when intent is unclear (sarcasm, idiom, metaphor).
- Human-audit routing (`quality_gates.py`): route whenever the majority-voted `Switch`
  is `"S"`, or the 3 runs disagree on `Switch`/`Escalation`.
