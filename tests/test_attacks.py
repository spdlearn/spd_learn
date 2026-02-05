"""Test the RiemannianPGDAttack attack on a simple model."""

import torch

from spd_learn.functional import spd_rpgd_attack
from spd_learn.functional.metrics.affine_invariant import airm_distance
from spd_learn.models import SPDNet


class TestRiemannianPGDAttack:
    """Test class for RiemannianPGDAttack."""

    def test_attack(self):
        # Simple test to check if the attack runs without errors
        model = SPDNet(input_type="cov", n_chans=3, n_outputs=2)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.eye(3).unsqueeze(0)  # Single SPD matrix
        y = torch.tensor([0])  # Dummy label
        epsilon = 0.2
        x_adv = spd_rpgd_attack(
            model,
            x,
            y,
            eps=epsilon,
            criterion=criterion,
            n_iterations=10,
            step_size=0.01,
        )
        assert x_adv.shape == x.shape, "Adversarial example has incorrect shape."

        # Check that the adversiarial example has an higher loss
        with torch.no_grad():
            output_clean = model(x)
            output_adv = model(x_adv)
            loss_clean = criterion(output_clean, y)
            loss_adv = criterion(output_adv, y)
            assert (
                loss_adv > loss_clean + 1e-4
            ), "Adversarial attack did not increase the loss sufficiently."
            assert (
                airm_distance(x, x_adv) <= epsilon + 1e-5
            ), "Adversarial example is outside the allowed perturbation radius."
