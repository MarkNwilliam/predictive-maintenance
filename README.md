<div align="center">

# Predictive Maintenance — 30-Day Failure Risk

**Failure probability in the next 30 days from motor temperature, vibration and running hours — a logistic regression fitted from scratch in pure Python**

Take the fleet of motors in a plant — a tablet press, a granulator, a water
system pump — and instead of reacting to breakdowns, score each machine's
probability of failing in the next 30 days and schedule maintenance *before*
the alarm.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![No deps](https://img.shields.io/badge/python-dependencies-none-blue)](#)
[![Stack](https://img.shields.io/badge/stack-HTML%20%2B%20CSS%20%2B%20JS-f59e0b)](#)
[![Tests](https://img.shields.io/badge/tests-13%20passing-green)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

</div>

---

## Why this exists

Most maintenance is still **run-to-failure** or **calendar-based**. Calendar
maintenance replaces parts too early; run-to-failure stops the line at the
worst moment. Condition-based maintenance uses the sensor data most machines
already generate — temperature, vibration, run time — to ask a precise question:
*what is the probability this machine fails in the next 30 days?* That is the
number that turns a maintenance cost into a planned one.

## The model

A **binary logistic regression** is fitted on labelled history:

| Feature | Meaning |
|---------|---------|
| `motor_temp`    | motor winding temperature (°C) |
| `vibration`     | velocity vibration (mm/s RMS) |
| `runtime_hours` | cumulative running hours |

```
P(failure in next 30 days) = sigmoid(w0 + w1·temp + w2·vib + w3·run)
```

- Features are **standardized** before fitting, so the coefficients are
  comparable (here: vibration + runtime dominate temperature).
- Trained end-to-end with **gradient descent written from scratch** — only the
  standard library. No pandas, no sklearn, no black box.
- Comes with a reproducible, physics-informed synthetic history generator so
  you can re-run the whole pipeline and see the math move.

Learned coefficients (from the demo fit, 240 readings):

```
intercept       -1.06
motor_temp      +0.87   per std dev
vibration       +1.36   per std dev
runtime_hours   +1.28   per std dev

Accuracy 80% · Precision 73% · Recall 68%
```

## The maintenance decision

| P(fail ≤ 30 days) | Action                    |
|-------------------|---------------------------|
| `< 25%` | Low — routine schedule      |
| `25–50%` | Moderate — monitor trend    |
| `50–75%` | High — inspect within the week |
| `≥ 75%` | Critical — plan maintenance now |

## What's in the repo

- **`predictive_maintenance.py`** — dependency-free library + CLI:
  `fit_logistic()` (gradient descent), `predict_proba()`, `predict()`,
  `risk_band()`, `evaluate()` (confusion matrix), `simulate_history()` and
  `failure_next_30d()` — the headline question in one call.
- **`test_predictive_maintenance.py`** — 13 unit tests: sigmoid properties,
  convergence on separable data, monotonicity, reproducibility of the
  generator, and a learnability check that the fitted model recovers > 78%
  accuracy on simulated data.
- **`index.html`** — interactive risk dashboard. Slides for temperature,
  vibration and runtime, a live probability meter, maintenance bands and
  pre-loaded scenarios. The embedded model parameters match the Python fit
  exactly.

## Quick start

```bash
# web dashboard
open index.html

# CLI — fits the model and shows risk probes
python3 predictive_maintenance.py

# library
python3 -c "
import predictive_maintenance as pm
hist = pm.simulate_history(seed=7)
model = pm.fit_logistic(hist['X'], hist['y'])
p = pm.failure_next_30d(model, 78.0, 4.2, 5200)
print(f'{p:.1%}  {pm.risk_band(p)}')"

# tests
python3 -m unittest test_predictive_maintenance -v
```

## Example output

```
Training set: 240 labelled sensor readings (85 failures within 30 days)
Accuracy  80.0%   Precision 73.4%   Recall 68.2%

Feature        coefficient (per standardized unit)
  motor_temp      +0.87
  vibration       +1.36
  runtime_hours   +1.28

Risk probes (current machine state -> 30-day failure probability)
  T=  65C  V=1.2 mm/s  run= 1500h  ->   0.7%  Low — routine schedule
  T=  78C  V=4.2 mm/s  run= 5200h  ->  52.6%  High — inspect within this week
  T=  88C  V=5.9 mm/s  run= 7600h  ->  96.6%  Critical — plan maintenance now
```

## Repository layout

```
predictive-maintenance/
├── index.html                      # interactive risk dashboard (open this)
├── predictive_maintenance.py       # fitted library + CLI
├── test_predictive_maintenance.py  # unit tests
└── README.md
```

## License

MIT.