"""Induced-norm linear layer with power iteration for Lipschitz constraint.

Port of PD-LTS PyTorch InducedNormLinear (models/layers/base/mixed_lipschitz.py).
Uses L2-norm (domain=2, codomain=2) power iteration to estimate spectral norm,
then soft-normalizes: W_out = W / max(1, sigma / coeff).
"""

from __future__ import annotations

import math

import jittor as jt
from jittor import nn


def _normalize_v(v: jt.Var) -> jt.Var:
    return v / (v.norm() + 1e-12)


def _normalize_u(u: jt.Var) -> jt.Var:
    return u / (u.norm() + 1e-12)


class InducedNormLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        coeff: float = 0.97,
        n_iterations: int | None = None,
        atol: float | None = 1e-3,
        rtol: float | None = 1e-3,
        zero_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol

        self.weight = jt.empty((out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if zero_init:
            self.weight = self.weight / 1000.0

        if bias:
            self.bias = jt.empty((out_features,))
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        u = jt.randn((out_features,))
        u = _normalize_u(u)
        v = jt.randn((in_features,))
        v = _normalize_v(v)

        self.u = u.stop_grad()
        self.v = v.stop_grad()
        self.scale = jt.zeros((1,)).stop_grad()

        self._init_power_iteration()

    def _init_power_iteration(self):
        with jt.no_grad():
            w = self.weight
            u = self.u
            v = self.v
            for _ in range(200):
                v_new = _normalize_v(jt.matmul(w.t(), u))
                u_new = _normalize_u(jt.matmul(w, v_new))
                v = v_new
                u = u_new
            self.u = u.stop_grad()
            self.v = v.stop_grad()
            sigma = (u * jt.matmul(w, v)).sum()
            self.scale = sigma.stop_grad()

    def compute_weight(self, update: bool = True) -> jt.Var:
        u = self.u
        v = self.v
        weight = self.weight

        if update:
            with jt.no_grad():
                n_iters = self.n_iterations
                if n_iters is None:
                    max_iters = 200
                else:
                    max_iters = n_iters

                for _ in range(max_iters):
                    if n_iters is None and self.atol is not None:
                        old_u = u.clone()
                        old_v = v.clone()

                    v_new = _normalize_v(jt.matmul(weight.t(), u))
                    u_new = _normalize_u(jt.matmul(weight, v_new))
                    v = v_new
                    u = u_new

                    if n_iters is None and self.atol is not None:
                        err_u = (u - old_u).norm() / math.sqrt(u.numel())
                        err_v = (v - old_v).norm() / math.sqrt(v.numel())
                        tol_u = self.atol + self.rtol * u.abs().max()
                        tol_v = self.atol + self.rtol * v.abs().max()
                        if float(err_u.item()) < float(tol_u.item()) and \
                           float(err_v.item()) < float(tol_v.item()):
                            break

                self.u = u.stop_grad()
                self.v = v.stop_grad()

        sigma = (u * jt.matmul(weight, v)).sum()
        with jt.no_grad():
            self.scale = sigma.stop_grad()

        factor = jt.maximum(jt.ones(1), sigma / self.coeff)
        w_normalized = weight / factor
        return w_normalized

    def execute(self, x: jt.Var) -> jt.Var:
        weight = self.compute_weight(update=self.is_training())
        out = jt.matmul(x, weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out

    def build_clone(self) -> nn.Linear:
        # CRITICAL: do NOT call .stop_grad() on self.bias / self.weight directly.
        # In Jittor, .stop_grad() modifies the in-place flag of the underlying
        # Var, which for nn.Parameter means the trainable parameter loses its
        # gradient permanently. Use .detach() to create an independent no-grad
        # copy without modifying the original. (PyTorch detach() and Jittor
        # detach() both create copies; Jittor stop_grad() does NOT.)
        with jt.no_grad():
            weight = self.compute_weight(update=False).detach()
            m = nn.Linear(self.in_features, self.out_features,
                          bias=(self.bias is not None))
            m.weight = weight.clone()
            if self.bias is not None:
                m.bias = self.bias.detach().clone()
            return m

    def build_jvp_net(self, x: jt.Var):
        with jt.no_grad():
            weight = self.compute_weight(update=False).detach()
            m = nn.Linear(self.in_features, self.out_features, bias=False)
            m.weight = weight.clone()
            y = self.execute(x).detach()
            return m, y
