"""Adapter, metrics and host-visible timing contracts."""
import unittest
import numpy as np

from evaluate_pi.benchmark import benchmark
from evaluate_pi.metrics import logits_to_probabilities, cross_entropy
from evaluate_pi.complexity import model_complexity


class EvaluationTests(unittest.TestCase):
    def test_benchmark_contract(self):
        class Adapter:
            calls = 0
            def predict(self, x):
                self.calls += 1
                assert x.shape == (1, 1, 300)
                return np.zeros((1, 5), dtype=np.float32)
        adapter = Adapter()
        summary, raw = benchmark(adapter, np.zeros((3, 1, 300), np.float32), warmup=2, runs=5)
        self.assertEqual(adapter.calls, 7)
        self.assertEqual(len(raw), 5)
        self.assertEqual(summary['batch_size'], 1)
        self.assertGreaterEqual(summary['mean_ms_per_sample'], 0)
        with self.assertRaises(ValueError):
            benchmark(adapter, np.zeros((3, 1, 300), np.float32), runs=0)

    def test_metric_and_count_contract(self):
        import torch
        from model import ECGClassifier
        from quantization.attention import explicit_model
        logits = np.array([[1000, 999, 998, 997, 996], [-1000, -999, -998, -997, -996]], dtype=np.float32)
        labels = np.array([0, 4])
        prob = logits_to_probabilities(logits)
        np.testing.assert_allclose(prob.sum(1), 1)
        self.assertAlmostEqual(cross_entropy(logits, labels), torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item(), places=5)
        original = ECGClassifier(transformer_blocks=1)
        model = explicit_model(original)
        counts = model_complexity(model, 300, quantized=False)
        self.assertEqual(counts['parameters_total'], sum(p.numel() for p in original.parameters()))
        self.assertEqual(counts['flops_equivalent_per_sample'], 2 * counts['macs_per_sample'])
        self.assertEqual(counts['macs_per_sample'], 7779840)
        self.assertEqual(counts['int8_ops_equivalent'], 0)
        qcounts = model_complexity(model, 300, quantized=True)
        self.assertEqual(qcounts['macs_per_sample'], counts['macs_per_sample'])
        self.assertGreater(qcounts['parameters_quantized'], 0)


if __name__ == '__main__':
    unittest.main()
