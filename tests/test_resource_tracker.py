import unittest
import os
import json
from monitoring import resource_tracker

class TestResourceTracker(unittest.TestCase):
    def test_get_resource_usage(self):
        usage = resource_tracker.get_resource_usage()
        self.assertIn("cpu_percent", usage)
        self.assertIn("memory", usage)
        self.assertIn("disk", usage)
        self.assertIn("net_io", usage)
        self.assertIn("timestamp", usage)

    def test_report_resource_usage(self):
        test_file = "test_resource_usage_report.json"
        if os.path.exists(test_file):
            os.remove(test_file)
        usage = resource_tracker.report_resource_usage(test_file)
        self.assertTrue(os.path.exists(test_file))
        with open(test_file) as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 1)
            data = json.loads(lines[-1])
            self.assertEqual(data["cpu_percent"], usage["cpu_percent"])
        os.remove(test_file)

if __name__ == "__main__":
    unittest.main()
