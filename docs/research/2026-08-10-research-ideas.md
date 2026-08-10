# Research idea candidates — toward a robotics-venue paper (drafted 2026-08-10)

Goal: a publishable, community-valuable paper THIS YEAR, building on what already
exists here (the eval harness, the thesis pilot data, the fed-mobile app, Maria's
Macs + iPhone, planned Pi/Jetson). Ranked by fit; decision points for Maria at the end.

## Venue reality check (checked 2026-08-10)

| Venue | Deadline | Fit |
|---|---|---|
| **ICRA 2027** (Seoul, May 2027) | **~Sep 17–22, 2026 — 5 weeks!** | Top venue; submission this year = goal met; VERY tight |
| **RA-L** (journal) | rolling — any time | Solid, respected, can also present at ICRA/IROS; safest "this year" target |
| **IROS 2027** | ~Mar 2027 | Comfortable timeline, submission not "this year" |
| **RO-MAN 2027** (human-robot interaction) | ~Feb–Mar 2027 | Best fit for the medical/assistive angle |
| **ICRA/IROS workshops** | ~2–3 months before event | Fast, forgiving, good for the benchmark-release idea |

## Idea 1 — "Can quantized on-device LLMs be trusted as robot command interpreters?"
**The main candidate.** Reframe the harness's core competence — structured
tool-calling fidelity under quantization × runtime-flag pressure, with joint
latency/memory/energy accounting — onto a ROBOT task suite: natural-language
commands → structured robot API calls (navigation, manipulation, home-assistant
actions). Tool call = robot action call; the schema mechanism is IDENTICAL to the
medical agent's, only the Pydantic models change.
- Contribution: first(?) systematic fidelity–latency–energy characterization of
  small quantized LLMs as command interpreters on real edge hardware (iPhone via
  fed-mobile telemetry, Pi 5/Jetson via llama-server), with actionable
  quantization/runtime guidance for roboticists.
- Reuses: ~everything. Needs: robot-command task suite (can be built on an existing
  benchmark subset + hand-authored home/care scenarios), n≥3 repeats + stats
  (statistical_analysis.py recovered), device fleet runs.
- Risk: reviewers ask "where is the robot?" — mitigate with one real embodied demo
  (even a simulated executor or a TurtleBot/rosbot run) or frame for RA-L/RO-MAN.

## Idea 2 — "On-device conversational patient interviews for assistive care robots"
**The medical use-case answer.** Maria's diabetes agent IS the dialog module of a
care/companion robot: a robot at home that interviews a diabetic patient daily —
privately, offline, on-device. Study: on-device vs cloud LLM for structured medical
data collection — task fidelity, interaction latency (what feels conversational),
energy per interview, and the privacy argument (health data never leaves the robot).
- Contribution: HRI-flavored systems study; strong societal angle; the fed-mobile
  app becomes the working artifact (reviewers love a running system).
- Venue: RO-MAN / ICORR / RA-L. Timeline fits "submit early 2027" better than Sept.
- This makes the medical app's role concrete: not the paper's core, but its
  demonstrator and one of two task suites.

## Idea 3 — "EdgeAgentBench: a reproducible harness for evaluating tool-calling LLMs on edge devices"
**The fallback / companion.** Open-source the rebuilt harness itself: manifest-
stamped, repeat-aware, device-tagged evaluation of structured tool-calling on edge
hardware, with the medical + robot task suites and baseline results across
quantizations. Community value is the point; workshops and demo tracks accept this
readily, and it multiplies citations of the main paper.
- Cheapest path to "something accepted this year" (workshop), and it IS the Part-D
  rebuild work anyway — the paper writes itself from the engineering.

## Idea 4 — Federated personalization of on-device agents (Maria's FL thread)
Real novelty, heavy lift (FL infrastructure that doesn't exist yet). NOT this year.
Park it as the follow-up paper; mention as future work in ideas 1–3.

## Recommended strategy (Claude's honest take)

**Primary: Idea 1 targeting RA-L, with an ICRA-2027 attempt only if the September
dry-run says the data will exist in time.** Five weeks means: task suite frozen by
week 1, experiments running by week 2, writing from week 3. Possible only because
the harness + pilot data exist — but one Mac + one iPhone is the whole fleet, so
scope ruthlessly (2 models × 4 quantizations × 4 runtime configs × n=5).
**Secondary: Idea 3 as an ICRA/IROS workshop paper** — near-free from the same work.
**Idea 2 becomes the RO-MAN submission in early 2027**, reusing the fleet + harness
with the medical suite — that's paper #2, and the answer to "should I use the
medical app as use case": yes, but as the SECOND paper's centerpiece, not the
first's (robotics venue #1 needs a robotics task).

## Decisions needed from Maria
1. Accept the two-paper strategy (robot-commands first, medical-HRI second)?
2. ICRA sprint: attempt the Sept deadline (intense 5 weeks) or go straight RA-L?
3. Hardware: buy a Pi 5 (~$80) now? It strengthens "edge" claims beyond Apple
   silicon and is the robotics-credible platform.
4. Robot task suite: build on which base? (I'll research candidates — e.g. BFCL
   subset + home-robot command sets — in the auto-research digest.)
