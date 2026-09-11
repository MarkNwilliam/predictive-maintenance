#!/usr/bin/env python3
"""Predictive maintenance — failure probability in the next 30 days.

A compact logistic-regression failure-risk model trained on historical sensor
readings from a rotating machine (e.g. a tablet press or granulator motor):

    features: motor temperature (C), vibration (mm/s), running hours
    label   : 1 if the machine failed (or required unscheduled repair)
              within the next 30 days, else 0

The whole library is pure Python + the standard library: the logistic
regression is fitted with gradient descent from scratch, so it runs anywhere
and is easy to audit — no pandas, no sklearn, no model black box.

Typical workflow in a plant:

    history = simulate_history(seed=7)          # labelled sensor history
    model   = fit_logistic(history["X"], history["y"])
    p       = failure_next_30d(model, 78.0, 4.2, 5200)
    # -> ~Probability of failure in next 30 days
"""
import math
import random

FEATURES = ("motor_temp", "vibration", "runtime_hours")


# --------------------------------------------------------------------------
# core math
# --------------------------------------------------------------------------
def sigmoid(z):
    """Logistic function, numerically stable for large |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def log_loss(y, p):
    """Cross-entropy loss of prediction p for a true label y in {0, 1}."""
    if not 0.0 < p < 1.0:
        raise ValueError("prediction must be strictly inside (0, 1)")
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def standardize(X):
    """Column means/stds of a list-of-lists dataset (feat-stable form)."""
    n = len(X)
    cols = len(X[0])
    means = [sum(r[c] for r in X) / n for c in range(cols)]
    stds = [math.sqrt(sum((r[c] - means[c]) ** 2 for r in X) / n)
            for c in range(cols)]
    stds = [s if s > 0 else 1.0 for s in stds]
    return means, stds


def fit_logistic(X, y, lr=0.2, epochs=600):
    """Fit a binary logistic regression by gradient descent.

    X : list of samples, each a list of numeric features
    y : list of labels 0/1, same length as X

    Returns the model dict {w0, w, means, stds, lr, epochs}. Features are
    standardized internally so coefficients are comparable.
    """
    if not X or len(X) != len(y):
        raise ValueError("X and y must be non-empty and the same length")
    n = len(X)
    cols = len(X[0])
    means, stds = standardize(X)
    Xs = [[ (r[c] - means[c]) / stds[c] for c in range(cols)] for r in X]

    w = [0.0] * (cols + 1)   # [bias, w1, w2, ...]
    for _ in range(epochs):
        grad = [0.0] * (cols + 1)
        for i in range(n):
            z = w[0] + sum(w[c + 1] * Xs[i][c] for c in range(cols))
            err = sigmoid(z) - y[i]
            grad[0] += err
            for c in range(cols):
                grad[c + 1] += err * Xs[i][c]
        for j in range(cols + 1):
            w[j] -= (lr / n) * grad[j]

    return {"w0": w[0], "w": w[1:], "means": means, "stds": stds,
            "features": FEATURES[:cols],
            "lr": lr, "epochs": epochs}


def decision_function(model, x):
    """Raw logit z for a single feature vector x."""
    z = model["w0"]
    for c in range(len(x)):
        z += model["w"][c] * (x[c] - model["means"][c]) / model["stds"][c]
    return z


def predict_proba(model, x):
    """P(failure in next 30 days) given standardised features."""
    return sigmoid(decision_function(model, x))


def predict(model, x, threshold=0.5):
    """Hard 0/1 prediction at the given decision threshold."""
    return 1 if predict_proba(model, x) >= threshold else 0


def risk_band(prob):
    """Maintenance decision band for a failure probability."""
    if prob >= 0.75:
        return "Critical — plan maintenance now"
    if prob >= 0.50:
        return "High — inspect within this week"
    if prob >= 0.25:
        return "Moderate — monitor trend"
    return "Low — routine schedule"


def failure_next_30d(model, motor_temp, vibration, runtime_hours):
    """Convenience wrapper matching the headline question (float prob)."""
    return predict_proba(model, [motor_temp, vibration, runtime_hours])


def evaluate(model, X, y, threshold=0.5):
    """Accuracy / precision / recall and the confusion counts."""
    if not X or len(X) != len(y):
        raise ValueError("X and y must be non-empty and the same length")
    tp = fp = tn = fn = 0
    for row, label in zip(X, y):
        pred = predict(model, row, threshold)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# --------------------------------------------------------------------------
# synthetic labelled history (physics-informed, reproducible)
# --------------------------------------------------------------------------
def simulate_history(n=240, seed=7):
    """Generate labelled sensor history for one machine.

    The underlying failure probability is a logistic function of temperature,
    vibration and runtime with realistic noise — the same shape the model must
    learn back.
    """
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        temp = rng.uniform(55, 95)        # motor winding temperature (C)
        vib = rng.uniform(0.6, 6.5)       # velocity (mm/s RMS)
        run = rng.uniform(300, 9000)      # cumulative running hours
        p = sigmoid(-8.0 + 0.045 * temp + 0.6 * vib + 0.00035 * run
                    + rng.gauss(0, 0.15))
        X.append([round(temp, 1), round(vib, 2), round(run, 0)])
        y.append(1 if rng.random() < p else 0)
    return {"X": X, "y": y}


# --------------------------------------------------------------------------
# demo run
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Predictive maintenance — logistic failure-risk model (pure Python)")
    print("=" * 62)
    hist = simulate_history(seed=7)
    X, y = hist["X"], hist["y"]
    model = fit_logistic(X, y)
    print(f"Training set: {len(y)} labelled sensor readings "
          f"({sum(y)} failures within 30 days)")
    metrics = evaluate(model, X, y)
    print(f"Accuracy  {metrics['accuracy']:.1%}   Precision "
          f"{metrics['precision']:.1%}   Recall {metrics['recall']:.1%}")
    print()
    print("Feature        coefficient (per standardized unit)")
    print(f"  intercept    {model['w0']:+.2f}")
    for name, coef in zip(model["features"], model["w"]):
        print(f"  {name:<15} {coef:+.2f}")
    print()
    print("Risk probes (current machine state -> 30-day failure probability)")
    for temp, vib, run in ((65, 1.2, 1500), (78, 4.2, 5200),
                           (88, 5.9, 7600)):
        p = failure_next_30d(model, temp, vib, run)
        print(f"  T={temp:>4}C  V={vib:>3.1f} mm/s  run={run:>5}h  "
              f"-> {p:>6.1%}  {risk_band(p)}")