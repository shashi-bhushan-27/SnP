# Historical-analog scenario back-test

Generated 2026-10-07T19:58:40+00:00 by `evaluation/backtest_analogs.py` on 81 curated events (2008-2025, `backend/config/analog_library.json`).

Each method predicts the held-out event's market reaction; errors are against what actually happened. P&L error = |P&L of the predicted shock - P&L of the realised shock| on the default $60M synthetic book.


## Horizon `trough` - leave-one-out (81 events scored)

| Method | P&L error | P&L rank corr. | Equity MAE | 10y MAE | Spread MAE | 10y sign correct |
|---|---|---|---|---|---|---|
| Hand-written matrix (scenarios.yaml) | $4.75M | 0.31 | 6.11 pp | 206.7 bp | 149.3 bp | 31% |
| Average of all past events | $0.39M | - | 2.82 pp | 10.9 bp | 9.1 bp | 66% |
| Average of same event type | $0.38M | 0.09 | 2.74 pp | 10.6 bp | 9.2 bp | 63% |
| Analog retrieval, bag-of-words | $0.46M | 0.06 | 3.29 pp | 13.5 bp | 10.6 bp | 53% |
| Analog retrieval, sentence embeddings (top-5) | $0.43M | 0.15 | 3.07 pp | 12.0 bp | 10.5 bp | 51% |
| Analog-weighted history (all events, type bonus) | $0.38M | 0.16 | 2.77 pp | 10.9 bp | 8.8 bp | 65% |

### Stress severity - horizon `trough`, leave-one-out

Coverage = share of events whose real loss was no worse than the scenario's; excess = how much more severe than reality the scenario was on average (capital held against losses that never came).

| Scenario rule | Coverage | Mean excess severity |
|---|---|---|
| history q20 (same type) | 77% | $0.26M |
| history q10 (same type) | 85% | $0.44M |
| history q10 (all events) | 89% | $0.49M |
| hand-written matrix | 95% | $4.72M |

## Horizon `trough` - time-respecting (66 events scored)

| Method | P&L error | P&L rank corr. | Equity MAE | 10y MAE | Spread MAE | 10y sign correct |
|---|---|---|---|---|---|---|
| Hand-written matrix (scenarios.yaml) | $4.73M | 0.44 | 6.36 pp | 213.7 bp | 138.9 bp | 35% |
| Average of all past events | $0.39M | - | 2.87 pp | 10.1 bp | 9.6 bp | 62% |
| Average of same event type | $0.38M | 0.23 | 2.88 pp | 11.4 bp | 10.0 bp | 53% |
| Analog retrieval, bag-of-words | $0.45M | 0.16 | 3.27 pp | 13.8 bp | 10.9 bp | 60% |
| Analog retrieval, sentence embeddings (top-5) | $0.42M | 0.16 | 3.10 pp | 12.2 bp | 9.8 bp | 62% |
| Analog-weighted history (all events, type bonus) | $0.38M | 0.23 | 2.82 pp | 10.9 bp | 9.2 bp | 60% |

### Stress severity - horizon `trough`, time-respecting

Coverage = share of events whose real loss was no worse than the scenario's; excess = how much more severe than reality the scenario was on average (capital held against losses that never came).

| Scenario rule | Coverage | Mean excess severity |
|---|---|---|
| history q20 (same type) | 77% | $0.23M |
| history q10 (same type) | 89% | $0.43M |
| history q10 (all events) | 89% | $0.50M |
| hand-written matrix | 94% | $4.69M |

### What history says vs the hand-written matrix - horizon `trough`

| Event type | Events | Median equity | Median 10y | 10y fell | Median Baa spread | Matrix (equity / 10y / spread) |
|---|---|---|---|---|---|---|
| Bankruptcy | 5 | -2.6% | -27 bp | 100% | +16 bp | -9% / +50 bp / +300 bp |
| Credit Event | 11 | -2.1% | -7 bp | 73% | +1 bp | -8% / +100 bp / +250 bp |
| Cybersecurity | 3 | -2.6% | +7 bp | 0% | +5 bp | none |
| Geopolitical | 16 | -0.7% | -7 bp | 81% | +3 bp | -10% / +200 bp / +150 bp |
| Macroeconomic | 29 | -1.9% | +1 bp | 48% | +4 bp | -7% / +300 bp / +100 bp |
| Market Shock | 15 | -4.8% | -7 bp | 67% | +5 bp | -12% / +200 bp / +200 bp |
| Product Launch | 1 | -1.5% | -10 bp | 100% | +2 bp | none |
| Regulatory | 1 | -0.8% | -9 bp | 100% | +1 bp | -6% / +100 bp / +100 bp |

## Horizon `1d` - leave-one-out (81 events scored)

| Method | P&L error | P&L rank corr. | Equity MAE | 10y MAE | Spread MAE | 10y sign correct |
|---|---|---|---|---|---|---|
| Hand-written matrix (scenarios.yaml) | $4.95M | 0.28 | 7.31 pp | 205.9 bp | 153.3 bp | 28% |
| Average of all past events | $0.23M | - | 2.12 pp | 8.4 bp | 4.6 bp | 70% |
| Average of same event type | $0.23M | 0.02 | 2.00 pp | 8.7 bp | 4.7 bp | 72% |
| Analog retrieval, bag-of-words | $0.29M | 0.07 | 2.48 pp | 10.0 bp | 6.0 bp | 61% |
| Analog retrieval, sentence embeddings (top-5) | $0.27M | 0.19 | 2.41 pp | 9.2 bp | 5.9 bp | 65% |
| Analog-weighted history (all events, type bonus) | $0.23M | 0.20 | 2.03 pp | 8.6 bp | 4.9 bp | 67% |

### Stress severity - horizon `1d`, leave-one-out

Coverage = share of events whose real loss was no worse than the scenario's; excess = how much more severe than reality the scenario was on average (capital held against losses that never came).

| Scenario rule | Coverage | Mean excess severity |
|---|---|---|
| history q20 (same type) | 77% | $0.20M |
| history q10 (same type) | 89% | $0.34M |
| history q10 (all events) | 90% | $0.32M |
| hand-written matrix | 95% | $4.94M |

## Horizon `1d` - time-respecting (66 events scored)

| Method | P&L error | P&L rank corr. | Equity MAE | 10y MAE | Spread MAE | 10y sign correct |
|---|---|---|---|---|---|---|
| Hand-written matrix (scenarios.yaml) | $4.90M | 0.44 | 7.31 pp | 212.9 bp | 142.3 bp | 32% |
| Average of all past events | $0.21M | - | 1.84 pp | 8.2 bp | 4.5 bp | 66% |
| Average of same event type | $0.22M | 0.14 | 2.02 pp | 8.8 bp | 4.5 bp | 64% |
| Analog retrieval, bag-of-words | $0.24M | 0.17 | 2.16 pp | 9.8 bp | 5.1 bp | 66% |
| Analog retrieval, sentence embeddings (top-5) | $0.22M | 0.27 | 1.97 pp | 8.5 bp | 4.5 bp | 66% |
| Analog-weighted history (all events, type bonus) | $0.20M | 0.30 | 1.81 pp | 8.3 bp | 4.3 bp | 64% |

### Stress severity - horizon `1d`, time-respecting

Coverage = share of events whose real loss was no worse than the scenario's; excess = how much more severe than reality the scenario was on average (capital held against losses that never came).

| Scenario rule | Coverage | Mean excess severity |
|---|---|---|
| history q20 (same type) | 73% | $0.16M |
| history q10 (same type) | 88% | $0.30M |
| history q10 (all events) | 91% | $0.35M |
| hand-written matrix | 94% | $4.89M |

### What history says vs the hand-written matrix - horizon `1d`

| Event type | Events | Median equity | Median 10y | 10y fell | Median Baa spread | Matrix (equity / 10y / spread) |
|---|---|---|---|---|---|---|
| Bankruptcy | 5 | -0.0% | -3 bp | 60% | +2 bp | -9% / +50 bp / +300 bp |
| Credit Event | 11 | -1.7% | -7 bp | 64% | +1 bp | -8% / +100 bp / +250 bp |
| Cybersecurity | 3 | -0.7% | +3 bp | 0% | +0 bp | none |
| Geopolitical | 16 | -0.3% | -6 bp | 69% | +2 bp | -10% / +200 bp / +150 bp |
| Macroeconomic | 29 | -1.1% | +0 bp | 48% | +1 bp | -7% / +300 bp / +100 bp |
| Market Shock | 15 | -3.0% | -4 bp | 80% | +4 bp | -12% / +200 bp / +200 bp |
| Product Launch | 1 | -1.5% | -10 bp | 100% | +2 bp | none |
| Regulatory | 1 | -0.5% | -4 bp | 100% | +1 bp | -6% / +100 bp / +100 bp |

Caveats: 81 events is a small sample, so differences of a few percent are noise; neighbours within 10 days of a held-out event are purged because they share its market window; the library was curated by hand (selection bias toward famous events); titles are written outcome-free so the held-out headline does not reveal its own market reaction. Sign accuracy counts only events where the 10y yield moved at least 5 bp.
