from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from read_capture import read_capture


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


class CaptureTests(unittest.TestCase):
    def test_event_order_and_dependency(self):
        self.assertEqual(np.__version__, "2.2.6")
        events = read_capture(CAPTURE)["events"]
        self.assertEqual([event[0] for event in events], ['clock', 'clock', 'chunk', 'chunk', 'safe', 'safe', 'preview', 'publish', 'clock', 'chunk', 'gap', 'chunk', 'safe', 'publish', 'clock', 'clock', 'preview'])

    def test_invalid_event_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"events":[[]]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "事件列表"):
                read_capture(path)

    def test_command_reads_the_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "read_capture", str(CAPTURE)], cwd=ROOT,
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(len(result.stdout.splitlines()), 17)

    def test_capture_geometry_and_values(self):
        capture = read_capture(CAPTURE)
        self.assertEqual({len(taps) for taps in capture["taps"].values()}, {len(capture["offsets"])})
        self.assertEqual(capture["events"][1][3], 0)
        self.assertGreater(capture["events"][8][3], 0)
        self.assertTrue(np.isclose(sum(capture["taps"]["2"]), 1))
