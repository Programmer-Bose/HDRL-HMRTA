import unittest
from types import SimpleNamespace

import torch

from ta_train import fleet_objectives, hp, scalarize


class FleetObjectivesTests(unittest.TestCase):
    def test_idle_robots_do_not_change_objectives_or_reward(self):
        env = SimpleNamespace(
            valid=torch.tensor([[True], [True], [False]]),
            pay=torch.tensor([[5.0], [5.0], [0.0]]),
            cap=torch.tensor([10.0, 10.0, 10.0]),
            T_used=torch.tensor([100.0, 200.0, 999.0]),
            E_used=torch.tensor([10.0, 30.0, 100.0]),
            usable=torch.tensor([100.0, 100.0, 100.0]),
        )

        makespan, mean_soc_drop, payload_var = fleet_objectives(env)
        weights = torch.tensor([0.0, 1.0, 0.0])

        self.assertEqual(makespan, 200.0)
        self.assertAlmostEqual(mean_soc_drop, 0.2)
        self.assertEqual(payload_var, 0.0)
        self.assertAlmostEqual(
            scalarize(makespan, mean_soc_drop, payload_var, 0, weights),
            -0.2,
        )

    def test_no_assigned_robots_returns_zero_objectives(self):
        env = SimpleNamespace(
            valid=torch.tensor([[False], [False]]),
            pay=torch.zeros((2, 1)),
            cap=torch.tensor([10.0, 10.0]),
            T_used=torch.zeros(2),
            E_used=torch.zeros(2),
            usable=torch.tensor([100.0, 100.0]),
        )

        self.assertEqual(fleet_objectives(env), (0.0, 0.0, 0.0))

    def test_unassigned_task_penalty_is_preserved(self):
        weights = torch.tensor([0.0, 1.0, 0.0])
        reward = scalarize(0.0, 0.2, 0.0, 1, weights)

        self.assertAlmostEqual(reward, -0.2 - hp.unassigned_penalty)


if __name__ == "__main__":
    unittest.main()
