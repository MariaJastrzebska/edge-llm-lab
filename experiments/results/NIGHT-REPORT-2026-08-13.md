# NIGHT REPORT — 2026-08-12/13 (Mac session): the short-forward-pass test

**Instrument:** MacBook Air, `MacBookAir10,1`, Apple M1, Darwin 25.3.0 arm64.
**Repo:** `706e745`. **llama.cpp:** `030ebb5` (unchanged across all nights).
**Predecessors:** `NIGHT-REPORT.md` (08-10/11, confounded), `NIGHT-REPORT-2026-08-12.md`.

**Headline: G1's mechanism prediction is supported.** At sub-25 ms/token, a
positive paired latency tax appears on **all four** sub-1B models, every
confidence interval excluding zero — where at 34–57 ms/token it was absent or
≤1.8 %. On `gemma3:1b` the tax is **+30 %**. The crossover is real, but it is
**not a clean function of decode speed**, and §4 explains why that matters.

---

## 1. What ran

| Experiment | Runs | Duration | Status |
|---|---|---|---|
| 005-g1-short-forward-pass | 480/480 | 3896 s | ok, **no model skipped** |
| 006-qwen3-adequate-budget | 40 | 4639 s | ok |

All four sub-1B models pulled and loaded cleanly: `qwen3:0.6b`,
`llama3.2:1b`, `gemma3:1b`, `qwen2.5:0.5b`. 35,909 power samples on 005.
Fresh timestamped dirs; earlier results untouched; no `*_expected.json` modified.
I stayed off the machine during measurement (all diagnostics below were run
after the queue exited, except credential checks done during the unmeasured
model-pull phase).

---

## 2. THE RESULT — paired tax vs decode speed

Paired within-run by (prompt, seed), `gbnf − unconstrained`, n=60 per model:

| model | unconstrained | paired diff | 95 % CI | relative |
|---|---|---|---|---|
| qwen2.5:0.5b | 11.65 ms/tok | +0.214 ms | [+0.130, +0.298] | **+1.84 %** |
| qwen3:0.6b | 13.71 | +0.331 ms | [+0.246, +0.417] | **+2.42 %** |
| gemma3:1b | 19.77 | +6.002 ms | [+5.335, +6.669] | **+30.36 %** |
| llama3.2:1b | 25.08 | +1.543 ms | [+1.159, +1.927] | **+6.15 %** |

### The crossover, all nights on one axis

| model (run) | decode speed | relative tax |
|---|---|---|
| qwen3:4b (003) | 56.82 ms/tok | −1.76 % |
| phi4-mini (003) | 52.71 | +1.51 % |
| phi4-mini (004) | 52.13 | +1.65 % |
| granite3.1 (003) | 34.64 | +0.34 % *(did not replicate)* |
| granite3.1 (004) | 34.69 | −0.12 % |
| **llama3.2:1b (005)** | **25.08** | **+6.15 %** |
| **gemma3:1b (005)** | **19.77** | **+30.36 %** |
| **qwen3:0.6b (005)** | **13.71** | **+2.42 %** |
| **qwen2.5:0.5b (005)** | **11.65** | **+1.84 %** |

Everything at ≥34 ms/token is within ±1.8 % and half of it fails to exclude
zero. Everything below 25 ms/token is positive with intervals excluding zero.
**That is the direction G1 predicts.**

---

## 3. But the mask cost is NOT a constant — and this complicates the story

The simplest reading of G1 is "a roughly fixed CPU mask cost stops hiding once
the forward pass gets short", which predicts a roughly *constant absolute*
per-token overhead. It is not constant. It varies **28×** across four models
that all decode in the same 11–25 ms band:

| model | absolute mask cost | decode speed |
|---|---|---|
| qwen2.5:0.5b | **+0.214 ms/token** | 11.65 ms/tok |
| qwen3:0.6b | +0.331 | 13.71 |
| llama3.2:1b | +1.543 | 25.08 |
| gemma3:1b | **+6.002** | 19.77 |

`gemma3:1b` pays 28× what `qwen2.5:0.5b` pays, while decoding *slower* (so
having *more* room to hide it). And the relative tax is not monotone in decode
speed: the two fastest models (11.65, 13.71 ms/tok) show the *smallest* relative
taxes, while `gemma3:1b` at 19.77 shows the largest by an order of magnitude.

**So decode speed alone does not predict the tax.** Something model-specific
dominates. A candidate worth testing on the Linux side is **vocabulary size** —
mask construction is O(vocab), and `gemma3` is known to use a much larger vocab
than the Qwen models — but I could not extract `n_vocab` from the server logs
tonight, so **this is an untested hypothesis, not a measurement.** Grammar-stack
depth per token and tokenizer fragmentation of JSON are equally plausible and
equally untested here.

---

## 4. ⚠ CONFOUND I CHECKED: generation length affects per-token time

Conditions do not generate the same number of tokens, and per-token decode time
rises with sequence length (KV cache growth). Measured directly, unconstrained
runs only, shortest third vs longest third:

| model | short | long | drift |
|---|---|---|---|
| qwen2.5:0.5b | 227 tok → 11.56 ms | 341 tok → 11.89 ms | +2.9 % |
| qwen3:0.6b | 383 → 13.63 | 628 → 13.79 | +1.2 % |
| gemma3:1b | 34 → 17.96 | 498 → 21.51 | +19.8 % |
| llama3.2:1b | 15 → 23.53 | 458 → 26.40 | +12.2 % |

Token counts per condition, and the resulting bias direction:

| model | tokens unc | tokens gbnf | bias on the tax | corrected estimate |
|---|---|---|---|---|
| qwen2.5:0.5b | 283.5 | 288.9 | negligible (+5 tok) | **~+1.8 %** |
| qwen3:0.6b | 504.2 | 268.6 | tax **understated** (gbnf 47 % shorter) | **≳+2.4 %** |
| gemma3:1b | 255.5 | 297.0 | ~0.32 ms of +6.00 ms | **~+29 %** |
| llama3.2:1b | 193.4 | 297.8 | ~0.68 ms of +1.54 ms | **~+3.4 %** |

**The conclusion survives.** `llama3.2:1b`'s tax roughly halves once length is
accounted for; `gemma3:1b`'s is essentially untouched; `qwen2.5:0.5b` needs no
correction at all and is the cleanest single number in this report — its
conditions generate within 2 % of the same token count. All four remain
positive.

These are back-of-envelope corrections from a two-bucket slope, not a fitted
model. A proper length-controlled regression is analysis-side work and I have
not done it. The per-run token counts are in `tidy.csv` for exactly that.

---

## 5. Experiment 006 — qwen3 with an adequate budget

`n_predict` raised 768 → **3072**, 20 prompts × 2 conditions.

| condition | ms/token | valid | tokens | truncated | J |
|---|---|---|---|---|---|
| unconstrained | 61.18 | **1/20** | 2930.8 | **17/20** | 1046.0 |
| gbnf | 56.92 | **20/20** | 371.1 | 0/20 | 179.5 |

paired: **−4.260 ms [−4.667, −3.854] = −6.96 %**

**Quadrupling the budget did not fix qwen3.** It still truncates 17 of 20 times,
now after ~2931 tokens of reasoning prose. Its 1046 J remains "cost of burning
the budget", and its 1/20 validity is still the truncation artifact, not a
decoding property.

The −6.96 % is the length confound of §4 in its most extreme form and **should
not be reported as constrained decoding being faster**: the unconstrained arm is
averaging per-token cost over a 2931-token sequence whose later tokens are
expensive, while the constrained arm finishes in 371. These two numbers are not
measuring the same thing. If qwen3 is wanted in the latency analysis at all, it
needs either thinking disabled or a length-matched comparison.

What 006 *does* establish cleanly is the validity/energy side: **20/20 vs 1/20,
and 179.5 J vs 1046.0 J** — a 5.8× energy difference driven entirely by the
unconstrained model's failure to terminate.

---

## 6. Validity in 005

Unconstrained scored **0/60 on all four models**; constrained scored 60/60,
60/60, 58/60 (`llama3.2:1b`) and 52/60 (`gemma3:1b`).

Pooled unconstrained failure modes: 117 × no JSON parse, 56 × type error,
31 × missing required property, 14 × `anyOf` failure, 4 × numeric. Sub-1B models
simply do not reliably produce this tool-call envelope unprompted — which is a
result in itself and is *not* the enum artifact (that was fixed in `b64b4ed`).

`gemma3:1b`'s 8 constrained failures are **all `stop_type=limit`** — truncation
at the 768-token cap, not grammar violations. Given it pays a 6 ms/token mask
cost, it is the model most likely to hit a cap. Worth a larger budget if
`gemma3:1b` matters.

---

## 7. ⚠ The Telegram night report did NOT send

Checked before the run, not after. **All three credential paths fail on this Mac:**

1. `gcloud` is not installed; the script's fallback
   `/usr/local/google-cloud-sdk/bin/gcloud` does not exist.
2. `~/.telegram.env` does not exist; `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
   are unset in the environment.
3. ADC exists at `~/.config/gcloud/application_default_credentials.json` but is
   **dead** — accessing either secret via the Python client fails with
   `invalid_grant`. The refresh token is expired or revoked.

`_notify_session_done` swallows all exceptions by design, so this fails
**silently**: the run looks fine and Maria receives nothing. The
`TELEGRAM_CHAT_ID` → `USER_ID` fix is correct but cannot help when the secret
cannot be read.

**Fix requires Maria:** either install the SDK and run
`gcloud auth application-default login`, or place `~/.telegram.env` with the two
values on this Mac. I did not go looking for a bot token.

**Second, independent flaw:** the hook attaches
`sorted(glob("NIGHT-REPORT-*.md"))[-1]` at the moment `run_experiments.py`
exits — *before* tonight's report exists. Even with working credentials it would
have attached the **previous** night's report as if it were current. The hook
should run after the report is written, or send text-only.

---

## 8. Answering the question that was asked

> Does a positive paired tax appear at 10–15 ms/token where it was absent at
> 37–66?

**Yes.** All four sub-1B models show it, all intervals exclude zero, and the
smallest of them (+1.8 % on the cleanest, length-matched model) is larger than
anything measured on the 34–57 ms/token models. This is **direct evidence for
G1's mechanism** on Apple Silicon: the CPU-side mask stops hiding as the forward
pass shortens.

**But the crossover is not a single number**, and G1 should not claim one. The
tax is not monotone in decode speed — `gemma3:1b` at 19.77 ms/tok pays +30 %
while `qwen3:0.6b` at 13.71 pays +2.4 %. Absolute mask cost varies 28× across
models in the same speed band, so **model identity dominates decode speed**.
Any mechanism claim has to explain that spread, and this rig has not identified
what drives it (§3 lists candidates; none tested).

The honest summary for the paper: *the tax is real and can be large on edge-class
models, it appears once decode drops below ~25 ms/token, and its size is
governed by something model-specific that is not yet identified.* That last
clause is the next experiment, not a caveat to bury.

**No go/no-go conclusion is drawn here.** That decision is Maria's and the Linux
side's.
