"""Decoding-error time schedule of Flow Map Language Models (arXiv:2602.16813).

Times are drawn and gridded in the schedule's uniform progress variable ``tau``,
which rescales flow time so that equal steps make equal progress in resolving
token identities. The schedule is a plain object rather than an ``nn.Module`` so
that casting the model's dtype never touches its float64 lookup tables.
"""

import math

import torch

__all__ = ["DecodingErrorSchedule", "decoding_error_rate"]


def decoding_error_rate(
    t: torch.Tensor, vocab_size: int, z_points: int = 2001, z_max: float = 10.0
) -> torch.Tensor:
    """``P_e(t)``: the chance the argmax of ``(1 - t) z + t e_y`` is not ``y``.

    Decoding is correct iff ``z_y + t / (1 - t)`` beats the largest of the other
    ``V - 1`` standard normals, so ``P_correct = E_z[Phi(z + c)^(V - 1)]`` with
    ``c = t / (1 - t)``, integrated by the trapezoid rule in float64.
    """
    t = t.to(torch.float64).clamp(0.0, 1.0)
    z = torch.linspace(-z_max, z_max, z_points, dtype=torch.float64, device=t.device)
    log_phi = -0.5 * z**2 - 0.5 * math.log(2 * math.pi)
    c = t / (1.0 - t).clamp_min(torch.finfo(torch.float64).tiny)
    c = torch.where(t >= 1.0, torch.full_like(c, float("inf")), c)
    log_integrand = (vocab_size - 1) * torch.special.log_ndtr(
        z[None, :] + c[:, None]
    ) + log_phi[None, :]
    p_correct = torch.trapezoid(log_integrand.exp(), z, dim=-1)
    return (1.0 - p_correct).clamp(0.0, 1.0)


class DecodingErrorSchedule:
    """``tau(t) = (P_e(0) - P_e(t)) / P_e(0)`` and its inverse, tabulated once.

    Training draws ``tau ~ U[0, 1]`` and maps it through ``t(tau)``; drafting in
    ``n`` steps uses the grid ``t(k / n)``.
    """

    def __init__(self, vocab_size: int, table_size: int = 4001):
        if vocab_size < 2:  # noqa: PLR2004
            raise ValueError("vocab_size must be at least 2")
        # Explicitly on the CPU: models are built under a meta-device context
        # when loaded with from_pretrained.
        t_table = torch.linspace(
            0.0, 1.0, table_size, dtype=torch.float64, device="cpu"
        )
        p_e = decoding_error_rate(t_table, vocab_size)
        p_e0 = 1.0 - 1.0 / vocab_size
        tau_table = ((p_e0 - p_e) / p_e0).clamp(0.0, 1.0)
        tau_table[0], tau_table[-1] = 0.0, 1.0
        tau_table = torch.cummax(tau_table, dim=0).values
        self._tables = {torch.device("cpu"): (t_table, tau_table)}

    def _tables_on(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        if device not in self._tables:
            t_table, tau_table = self._tables[torch.device("cpu")]
            self._tables[device] = (t_table.to(device), tau_table.to(device))
        return self._tables[device]

    @staticmethod
    def _interp(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        x64 = x.to(torch.float64).clamp(xs[0], xs[-1])
        idx = torch.searchsorted(xs, x64, right=True).clamp(1, xs.numel() - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        w = torch.where(x1 > x0, (x64 - x0) / (x1 - x0), torch.zeros_like(x64))
        return (y0 + w * (y1 - y0)).float()

    def tau(self, t: torch.Tensor) -> torch.Tensor:
        t_table, tau_table = self._tables_on(t.device)
        return self._interp(t, t_table, tau_table)

    def t(self, tau: torch.Tensor) -> torch.Tensor:
        t_table, tau_table = self._tables_on(tau.device)
        return self._interp(tau, tau_table, t_table)

    def sample(self, shape, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(tau, t)`` with ``tau ~ U[0, 1]``."""
        tau = torch.rand(tuple(shape), dtype=torch.float64, device=device)
        return tau.float(), self.t(tau)

    def grid(self, num_steps: int, device) -> torch.Tensor:
        """``t(k / num_steps)`` for ``k = 0..num_steps``, ends set to exactly 0, 1."""
        tau = torch.linspace(
            0.0, 1.0, num_steps + 1, dtype=torch.float64, device=device
        )
        grid = self.t(tau)
        grid[0], grid[-1] = 0.0, 1.0
        return grid
