from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from evaluate_int8_runtime import (
    audit_quantized_graph,
    percentile,
    select_quantized_engine,
)


class Int8RuntimeTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 3.0], 0.95), 2.9)

    def test_auto_engine_is_supported(self) -> None:
        engine = select_quantized_engine("auto")
        self.assertIn(engine, torch.backends.quantized.supported_engines)

    def test_audit_rejects_float_convolution(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3))
        audit = audit_quantized_graph(
            model,
            original_conv_count=1,
            original_linear_count=0,
            original_residual_add_count=0,
        )
        self.assertFalse(audit["strict_integer_core"])
        self.assertEqual(audit["float_conv_names"], ["0"])


if __name__ == "__main__":
    unittest.main()
