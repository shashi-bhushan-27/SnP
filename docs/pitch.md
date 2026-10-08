# Pitch: 7 slides and a 5-minute demo

Every number below comes from a script in this repository (`evaluation/run_eval.py`,
`evaluation/backtest_analogs.py`) or from a live run; re-run them before presenting if anything changed.

## Slides

**1. Problem.** Markets react to text before they react to numbers. A bank needs to know, quickly and
explainably, what a breaking event could cost its portfolio. Today's stress scenarios are typed in by hand.

**2. Solution: PRISM.** News (NewsAPI, GDELT) and social posts -> NLP Risk Engine -> structured signal
(entity, event type, sentiment, impact, confidence) -> automatic, history-calibrated stress test -> dashboard.

**3. Architecture.** ETL with three source adapters, idempotent loads, run reports; the engine as a pipeline of
swappable components behind protocols; Module B as a plug-in consumer; one composition root; a test that
enforces the module boundaries. 220+ tests, CI on every push.

**4. The NLP Risk Engine.** FinBERT sentiment; rules + sentence-embedding event classifier with calibrated
abstention; dictionary + spaCy entities; explainable impact (every point accounted for); confidence on every
signal. Measured: event macro-F1 0.77 (hybrid, right 83% of the time it answers) vs 0.78 embeddings (63%) vs 0.62 rules on 1,099 labelled tweets; sentiment macro-F1
0.66 (FinBERT) vs 0.60 (lexicon) on 2,388. Real-data error analysis: 130 -> 64 signals on 119 live GDELT
articles after the relevance fixes; 0 false stress triggers on 50 live NewsAPI articles (was 1).

**5. The novel part: history-calibrated stress.** Hand-written shocks assume yields rise in a crisis; in 81 real
events the 10-year yield fell after 100% of bankruptcies and 81% of geopolitical shocks. PRISM shocks the book
with what markets actually did after past events of the same kind, and shows the closest analogs.

**6. Results.** Back-test on 81 events (purged, and time-respecting): history at 1-in-10 covers 85-89% of real
outcomes with ~$0.4M excess; the hand-written matrix covers 95% with ~$4.7M excess, i.e. about ten times the
over-reserving. The worst real outcome for this book in 18 years is -$2.72M (March 2020); every hand-written
scenario is harsher. Honest negative: retrieval by meaning does not beat the type average for point prediction.

**7. Value and next steps.** Faster, explainable, evidence-based risk response; scenarios a risk committee can
audit ("this is what happened after Lehman"). Next: macro-direction rule for FinBERT, a larger event library,
sector-level shocks, more factors.

## 5-minute demo

| Time | Show | Say |
|---|---|---|
| 0:00 | Dashboard, empty | "Text in, portfolio risk out." |
| 0:20 | Auto-stream replay | Signals appear with entity, event, sentiment, impact, confidence. |
| 1:00 | Click one signal | "Why this score?" breakdown; event evidence; corroboration. |
| 1:40 | A high-impact negative event arrives | The stress test fires automatically (confirmed by a trusted source). |
| 2:20 | Stress panel | Basis event ("what markets did after ..."), average past outcome, the distribution, the analogs. |
| 3:10 | "Compare with the hand-written matrix" | -$5.2M vs about -$0.5M; the rates leg had the wrong sign; Treasuries hedge in a flight to safety. |
| 3:50 | Evidence table | Yields fell after 100% of bankruptcies and 81% of geopolitical shocks. |
| 4:20 | Back-test slide | Coverage vs excess; honest negative result. |
| 4:50 | Close | "Not a sentiment dashboard: an evidence-based risk decision engine." |

Fallbacks: replay mode needs no network; the models are cached locally; record a video of the run beforehand.
