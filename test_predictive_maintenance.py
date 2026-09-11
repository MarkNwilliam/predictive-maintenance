import math
import unittest

import predictive_maintenance as pm


class SigmoidAndLossTests(unittest.TestCase):
    def test_sigmoid_edges(self):
        self.assertAlmostEqual(pm.sigmoid(0), 0.5)
        self.assertAlmostEqual(pm.sigmoid(1000), 1.0, places=4)
        self.assertAlmostEqual(pm.sigmoid(-1000), 0.0, places=4)

    def test_sigmoid_symmetry(self):
        self.assertAlmostEqual(pm.sigmoid(2) + pm.sigmoid(-2), 1.0)

    def test_log_loss(self):
        self.assertAlmostEqual(pm.log_loss(1, 0.9), -math.log(0.9))
        with self.assertRaises(ValueError):
            pm.log_loss(1, 0.0)


class StandardizeTests(unittest.TestCase):
    def test_standardize(self):
        means, stds = pm.standardize([[10, 0], [20, 10]])
        self.assertAlmostEqual(means[0], 15.0)
        self.assertAlmostEqual(stds[0], 5.0)
        self.assertAlmostEqual(means[1], 5.0)

    def test_zero_std_is_safe(self):
        _, stds = pm.standardize([[5, 1], [5, 3]])
        self.assertEqual(stds[0], 1.0)


class ModelTests(unittest.TestCase):
    def _separable_data(self):
        # clearly separable: high temp+vib => failure
        X = [[60 + i, 5 + i, 1000 + 100 * i] for i in range(20)]
        X += [[30 + i % 5, 0.5 + i / 10, 400 + i] for i in range(20)]
        y = [1] * 20 + [0] * 20
        return X, y

    def test_fit_converges_on_separable_data(self):
        X, y = self._separable_data()
        model = pm.fit_logistic(X, y, lr=0.3, epochs=1500)
        # all high-risk samples predicted 1, all low-risk predicted 0
        self.assertEqual(pm.predict(model, [70, 6, 2000]), 1)
        self.assertEqual(pm.predict(model, [35, 0.8, 500]), 0)

    def test_proba_monotonic_with_features(self):
        X, y = self._separable_data()
        model = pm.fit_logistic(X, y, lr=0.3, epochs=1500)
        p_low = pm.predict_proba(model, [30, 0.5, 400])
        p_high = pm.predict_proba(model, [80, 7, 8000])
        self.assertGreater(p_high, p_low)
        self.assertTrue(0.0 <= p_low <= 1.0)

    def test_input_guards(self):
        with self.assertRaises(ValueError):
            pm.fit_logistic([], [])
        with self.assertRaises(ValueError):
            pm.evaluate({}, [], [])


class BandAndEvaluationTests(unittest.TestCase):
    def test_risk_bands(self):
        self.assertEqual(pm.risk_band(0.00), "Low — routine schedule")
        self.assertEqual(pm.risk_band(0.30), "Moderate — monitor trend")
        self.assertEqual(pm.risk_band(0.60), "High — inspect within this week")
        self.assertEqual(pm.risk_band(0.90), "Critical — plan maintenance now")

    def test_evaluate_confusion(self):
        model = {"w0": 0.0, "w": [1.0], "means": [0.0], "stds": [1.0]}
        X = [[-2.0], [-1.0], [1.0], [2.0]]
        y = [0, 0, 1, 1]
        m = pm.evaluate(model, X, y)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 2)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)


class SimulationAndCLITests(unittest.TestCase):
    def test_simulate_reproducible(self):
        a = pm.simulate_history(n=50, seed=7)
        b = pm.simulate_history(n=50, seed=7)
        self.assertEqual(a["X"], b["X"])
        self.assertEqual(a["y"], b["y"])
        self.assertEqual(len(a["X"]), 50)
        self.assertEqual(len(a["y"]), 50)

    def test_simulation_learnable(self):
        hist = pm.simulate_history(seed=7)
        model = pm.fit_logistic(hist["X"], hist["y"])
        m = pm.evaluate(model, hist["X"], hist["y"])
        self.assertGreaterEqual(m["accuracy"], 0.78)
        self.assertGreater(m["recall"], 0.6)

    def test_headline_question(self):
        hist = pm.simulate_history(seed=7)
        model = pm.fit_logistic(hist["X"], hist["y"])
        p = pm.failure_next_30d(model, 78.0, 4.2, 5200)
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertIn(pm.risk_band(p), {
            pm.risk_band(0.0), pm.risk_band(0.3), pm.risk_band(0.6),
            pm.risk_band(0.9)})
        # higher load should always push risk up on this fitted model
        self.assertGreater(
            pm.failure_next_30d(model, 92, 6.8, 8500),
            pm.failure_next_30d(model, 60, 0.9, 800))


if __name__ == "__main__":
    unittest.main()