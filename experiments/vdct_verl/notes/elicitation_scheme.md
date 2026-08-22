# VDCT elicitation scheme (Phase 0 note)

Decision note fixing the verbalized-distribution elicitation format for VDCT,
per the plan's Phase 0. One page: the chosen scheme, why, and the failure
modes the diagnostics must watch.

## Chosen scheme

CoT first (decided upstream), then a single fenced block that distributes
probability over **all** options of the question:

```
<distribution>
A: 0.62
B: 0.21
C: 0.09
D: 0.08
</distribution>
```

Exact instruction text and the strict parser live in
`recipe/vdct/vdct_elicitation.py` (single source of truth for both the
dataset builder and the agent loop). Parser contract: last
`<distribution>` block wins (the CoT may mention the tag); every option
label exactly once and no others; values non-negative; either unit
probabilities summing to 1 or percentages summing to 100 (auto-detected from
the total, `%` suffixes allowed), within ±5% tolerance; then
renormalize → floor at 1e-3 → renormalize.

## Why single-shot full-distribution (and not the alternatives)

- **Top-k guesses with probabilities** (the best performer in Tian et al.
  2023 for open-ended QA) leaves options uncovered; the JS and log-score
  terms need mass on every option, so uncovered options would force an
  imputation rule — more parser policy, no calibration benefit in the fixed
  small-K MCQ setting where "all options" and "top-k" nearly coincide.
- **Multi-sample summarization** conflicts with the method: each rollout
  must state its own distribution for the GRPO group to carry per-rollout
  reward.
- Verbalized numerical probabilities from post-RLHF models are better
  calibrated than their token logits (Tian et al. 2023, ~50% relative ECE
  reduction; Lin/Hilton/Evans 2022 for verbalized uncertainty generally),
  and models generate full distributions over categorical labels robustly
  (Liu et al. 2024, "Calibrating Verbalized Probabilities", arXiv
  2410.06707). No off-the-shelf scheme with demonstrated MCQ
  *distribution* calibration mandates a different surface format, so the
  simplest parseable one wins.
- Self-prediction of own behavior is trainable but weak zero-shot (Binder
  et al. 2024, "Looking Inward", arXiv 2410.13787: privileged self-access
  appears after finetuning on simple properties). VDCT's proper-scoring
  term is exactly that training signal, delivered by RL instead of SFT —
  expect poor base-model calibration against own-answer distributions
  (Phase 1.6a measures it) rather than treating it as a blocker.

## Pathologies for the diagnostics to watch

1. **Round-number clustering** — verbalized values cluster at multiples of
   5/10 and at 80/90/95/100% (Xiong et al. 2023 elicitation comparisons).
   Expect coarse distributions; fine for training (graded reward still
   orders them), but reliability plots should bin accordingly.
2. **Post-RLHF overconfidence / near-one-hot restating** — the model may
   restate its post-CoT conclusion as ~1.0 on one option (plan §6c). Watch
   mean entropy; if base distributions are near-one-hot, the method's case
   rests on graded reward + full-distribution coverage, not variance
   reduction — record it, per the plan.
3. **Re-softmax / scale distortion** (Liu et al. 2024) — stated values may
   behave like re-softmaxed logits, compressing extremes. Reliability
   curves (Phase 1.6a) catch this; no parser-side correction.
4. **Option-order anchoring** — position bias toward early labels. The
   audit set's empirical answer distributions share the same option order,
   so ECE comparisons are position-matched; cue-invariance diagnostics
   break out mass shifts on/off the cued option instead.
5. **Format non-compliance under RL drift** — the smoke gates ≥95% parse
   compliance *during* training, not just at init; unparseable
   distributions get worst-case reward, so compliance is trained.

## Sources

- Tian et al. 2023, *Just Ask for Calibration* — arXiv 2305.14975 / EMNLP 2023.
- Lin, Hilton, Evans 2022, *Teaching Models to Express Their Uncertainty in Words* — arXiv 2205.14334.
- Kadavath et al. 2022, *Language Models (Mostly) Know What They Know* — arXiv 2207.05221.
- Xiong et al. 2023, *Can LLMs Express Their Uncertainty?* — arXiv 2306.13063.
- Liu et al. 2024, *Calibrating Verbalized Probabilities for Large Language Models* — arXiv 2410.06707.
- Binder et al. 2024, *Looking Inward* — arXiv 2410.13787.
- LACIE (Stengel-Eskin et al. 2024) and SaySelf (Xu et al. 2024) — calibration finetuning lines; SFT-based, not adopted (VDCT trains calibration through the proper-scoring reward instead).
