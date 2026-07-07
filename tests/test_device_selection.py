import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.device import DEVICE_CHOICES, resolve_device


class DeviceSelectionTest(unittest.TestCase):
    def test_resolve_cpu(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_resolve_auto(self):
        self.assertIn(resolve_device("auto").type, {"cuda", "mps", "cpu"})

    def test_unknown_device_raises(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")

    def test_device_choices(self):
        self.assertEqual(DEVICE_CHOICES, ("auto", "cuda", "mps", "cpu"))


if __name__ == "__main__":
    unittest.main()
