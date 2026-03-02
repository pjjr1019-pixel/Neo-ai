import unittest
from scheduler.cost_scheduler import Job, CostScheduler

class TestCostScheduler(unittest.TestCase):
    def test_add_and_get_job(self):
        scheduler = CostScheduler()
        job1 = Job("job1", priority=1, estimated_cost=10)
        job2 = Job("job2", priority=2, estimated_cost=5)
        scheduler.add_job(job1)
        scheduler.add_job(job2)
        next_job = scheduler.get_next_job()
        self.assertEqual(next_job.job_id, "job2")  # Lower cost, higher priority
        self.assertEqual(scheduler.get_next_job().job_id, "job1")
        self.assertIsNone(scheduler.get_next_job())

    def test_optimize_schedule(self):
        scheduler = CostScheduler()
        usage = scheduler.optimize_schedule()
        self.assertIn("cpu_percent", usage)
        self.assertIn("memory", usage)

if __name__ == "__main__":
    unittest.main()
