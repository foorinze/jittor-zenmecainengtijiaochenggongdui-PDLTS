"""Fixed-point solvers and implicit differentiation for iMonotoneBlock.

Port of PD-LTS PyTorch solvers.py:
- find_fixed_point: Banach iteration x = y - G(x)
- root_find: no-grad wrapper (equivalent to RootFind autograd Function)
- monotone_block_backward: implicit differentiation (I + J_G^T)^{-1} @ grad

In PyTorch these use torch.autograd.Function. In Jittor we use stop_grad()
and jt.grad() to achieve the same gradient routing.
"""

from __future__ import annotations

import logging

import jittor as jt

logger = logging.getLogger(__name__)


def find_fixed_point(g_fn, y: jt.Var, atol: float = 1e-6, rtol: float = 1e-6,
                     max_iter: int = 10) -> jt.Var:
    """Solve x = y - G(x) via Banach iteration (no gradient)."""
    with jt.no_grad():
        x = y - g_fn(y)
        for i in range(max_iter):
            x_prev = x
            x = y - g_fn(x)
            tol = atol + y.abs() * rtol
            if jt.all((x - x_prev) ** 2 / (tol + 1e-12) < 1.0):
                break
    return x


def find_fixed_point_noaccel(f_fn, x0: jt.Var, threshold: int = 1000,
                             eps: float = 1e-3) -> jt.Var:
    """Damped fixed-point iteration with adaptive step size (no gradient)."""
    with jt.no_grad():
        B, N, C = x0.shape
        b_shape = (B, N, 1)
        alpha = 0.5 * jt.ones(b_shape)
        x = (1 - alpha) * x0 + alpha * f_fn(x0)
        tol = eps + eps * x0.abs()

        best_err = 1e9 * jt.ones(b_shape)
        best_iter = jt.zeros(b_shape, dtype='int64')

        for i in range(threshold):
            fx = f_fn(x)
            err_values = (fx - x).abs() / (tol + 1e-12)
            cur_err = err_values.max(dim=2, keepdims=True)

            if jt.all(cur_err < 1.0):
                break

            stall_mask = (cur_err >= best_err) & (jt.array(i) >= best_iter + 30)
            alpha = jt.where(stall_mask, alpha * 0.9, alpha)
            alpha = jt.maximum(alpha, jt.array(0.1))

            update_mask = (cur_err < best_err) | (jt.array(i) >= best_iter + 30)
            best_iter = jt.where(update_mask, jt.full(b_shape, i, dtype='int64'), best_iter)
            best_err = jt.minimum(best_err, cur_err)

            x = (1 - alpha) * x + alpha * fx

        return x


def root_find(g_fn, y: jt.Var) -> jt.Var:
    """No-grad fixed-point solve. Equivalent to PyTorch RootFind.apply().

    Returns detached result - no gradient flows through the solve itself.
    Gradient routing is handled by monotone_block_backward.

    Use .detach() not .stop_grad() — Jittor's stop_grad mutates the var in
    place, which silently corrupts upstream gradient paths if the result is
    not a fresh copy.
    """
    result = find_fixed_point(g_fn, y)
    return result.detach()


def monotone_block_backward_solve(g_fn, w_proxy: jt.Var, x: jt.Var,
                                  grad_output: jt.Var) -> jt.Var:
    """Solve implicit differentiation: dl/du = grad @ (I + J_G^T)^{-1}.

    This is the backward pass of MonotoneBlockBackward. It solves:
        (I + J_G^T) @ dl_dh = grad_output
    via fixed-point iteration on:
        h(z) = grad_output - J_G^T @ z

    Args:
        g_fn: the Lipschitz network G
        w_proxy: the point at which to evaluate Jacobian (saved from forward)
        x: the input (saved from forward)
        grad_output: incoming gradient

    Returns:
        dl_dh: solved implicit gradient
    """
    w = w_proxy.stop_grad()
    w.start_grad()

    def h_fn(z: jt.Var) -> jt.Var:
        gw = g_fn(w)
        jvp = jt.grad(gw, w, z)
        return grad_output - jvp

    dl_dh = root_find(lambda z: z - h_fn(z), grad_output)
    return dl_dh
