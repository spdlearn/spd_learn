import torch

from spd_learn.functional import matrix_sqrt_inv, spd_egrad2rgrad

from .metrics.affine_invariant import airm_distance, exp_map_airm, log_map_airm


def spd_rpgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    criterion: torch.nn.Module,
    n_iterations: int = 100,
    step_size: float = 0.01,
):
    r"""Riemannian Projected Gradient Descent attack on the SPD manifold under AIRM metric.

    This attack generates adversarial examples by performing gradient ascent on
    the loss function while constraining perturbations to lie within a geodesic
    ball of radius `eps` on the Riemannian manifold of SPD matrices :cite:p:`timoz2026riemannian`.

    Problem description: Given an input SPD matrix :math:`X \in \mathcal{S}^{++}_n` with
    its true label :math:`y`, the goal is to find an adversarial example
    :math:`X_{adv}` such that:

    .. math::

        \max_{X_{adv} \in \mathcal{S}^{++}_n} \mathcal{L}(f(X_{adv}), y)
        \quad \text{s.t.} \quad d_{AIRM}(X, X_{adv}) \leq \epsilon

    The algorithm outline:

    1. Initialize the adversarial example as the original input.
    2. For a fixed number of iterations:

    a. Compute the gradient of the loss w.r.t. the current adversarial example.
    b. Convert the Euclidean gradient to a Riemannian gradient.
    c. Take a step along the Riemannian gradient.
    d. Project back onto the geodesic ball of radius `eps`.

    Parameters
    ----------
    model : torch.nn.Module or object
        The model to attack. Should accept SPD matrices as input. If an object
        with a ``module_`` attribute (e.g. skorch) is passed, the underlying
        module is used.
    eps : float, default=0.2
        Maximum AIRM distance for adversarial perturbations.
    criterion : callable, optional
        Loss function to maximize. If None, uses `torch.nn.CrossEntropyLoss()`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.attacks import RiemannianPGDAttack
    >>> from spd_learn.models import SPDNet
    >>> model = SPDNet(n_channels=3, n_classes=2)
    >>> attack = RiemannianPGDAttack(
    ...     estimator=model,
    ...     eps=0.1,
    ...     criterion=torch.nn.CrossEntropyLoss()
    ... )
    >>> x = torch.randn(5, 3, 10, 10)
    >>> x = x @ x.transpose(-1, -2) + torch.eye(10)
    >>> y = torch.randint(0, 2, (5,))
    >>> x_adv = attack.generate(x, y, n_iterations=50, step_size=0.01)

    Notes
    -----
    The stepsize and number of iterations should be chosen carefully to balance
    attack strength and computational cost.


    See Also
    --------
    :func:`geodesic_distance_spdairm` : Distance under affine-invariant metric.
    """

    if not isinstance(model, torch.nn.Module):
        if hasattr(model, "module_"):
            model = model.module_
        else:
            raise TypeError(
                f"model must be a torch.nn.Module or have a 'module_' attribute, "
                f"got {type(model).__name__}"
            )

    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(n_iterations):
        outputs = model(x_adv)
        loss = criterion(outputs, y)
        euc_grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]

        with torch.no_grad():
            rgrad = spd_egrad2rgrad(x_adv, euc_grad)

            # Normalize the Riemannian gradient to have unit norm by AIRM
            _, x_inv_sqrt = matrix_sqrt_inv.apply(x_adv)
            v_orth = x_inv_sqrt @ rgrad @ x_inv_sqrt
            v_norm = torch.norm(v_orth, dim=(-2, -1), p="fro", keepdim=True)
            v_norm = torch.clamp(v_norm, min=1e-10)  # Avoid division by zero
            rgrad = rgrad / v_norm

            x_new = exp_map_airm(x_adv, rgrad, step_size)

            # Project back in the SPD ball wrt. the Riemannian metric
            x_adv = _project_to_spd_ball(x, x_new, eps)

        x_adv.requires_grad_(True)

    return x_adv.detach()


def _project_to_spd_ball(x0: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    r"""Project onto geodesic ball under AIRM metric.

    Projects `x` onto the closed geodesic ball:

    .. math::

        \mathcal{B}(x_0, \epsilon) = \{Y \in \mathcal{P}(n) : d_{AIRM}(x_0, Y) \leq \epsilon\}

    If :math:`d_{AIRM}(x_0, x) \leq \epsilon`, returns `x` unchanged.
    Otherwise, scales the tangent vector :math:`V = \log_{x_0}(x)` and
    returns :math:`\exp_{x_0}(\frac{\epsilon}{d} V)` where :math:`d = d_{AIRM}(x_0, x)`.

    Parameters
    ----------
    x0 : torch.Tensor
        Center of the geodesic ball, shape `(..., n, n)`.
    x : torch.Tensor
        Point to project, shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Projected point satisfying :math:`d_{AIRM}(x_0, \text{result}) \leq \epsilon`,
        with the same shape as `x`.

    Notes
    -----
    This projection is based on the exponential and logarithmic maps of the
    AIRM metric, ensuring the result lies on the geodesic from `x0` to `x`.
    """
    dist = airm_distance(x0, x)
    eps_t = torch.as_tensor(eps, dtype=dist.dtype, device=dist.device)

    if dist.ndim == 0:
        if dist > eps_t:
            v = log_map_airm(x0, x)
            v = (eps_t / dist) * v
            x = exp_map_airm(x0, v)
        return x

    mask = dist > eps_t
    if mask.any():
        v = log_map_airm(x0, x)
        scale = torch.where(mask, eps_t / dist, torch.ones_like(dist))
        v = v * scale[..., None, None]
        x = torch.where(mask[..., None, None], exp_map_airm(x0, v), x)

    return x
