"""Lipschitz-constrained fully-connected network for iMonotoneBlock.

Port of PD-LTS PyTorch FCNet (models/model_heavy/deflow.py:340-448).

Activation function: PyTorch reference defaults to ELU (deflow.py:56
`activation_fn='elu'`). Source-aligned default: ELU. Swish kept as ablation
option. ELU is strictly 1-Lipschitz on its derivative, giving the FCNet a
total Lipschitz bound of `coeff^(nhidden+1)` (assuming InducedNormLinear
constrains each layer to <= coeff). For coeff=0.9, nhidden=2 → 0.9^3 ≈ 0.729.

JVP path: ELU derivative is `1 if x > 0 else exp(x)`, used as the per-layer
multiplier in `_FCNetJVP.execute`. Swish derivative kept as fallback.
"""

from __future__ import annotations

import jittor as jt
from jittor import nn

from .spectral_norm import InducedNormLinear


def _swish(x: jt.Var) -> jt.Var:
    return x * jt.sigmoid(x)


def _swish_derivative(x: jt.Var) -> jt.Var:
    """d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
    s = jt.sigmoid(x)
    return s + x * s * (1.0 - s)


def _elu(x: jt.Var, alpha: float = 1.0) -> jt.Var:
    """ELU: x if x >= 0 else alpha * (exp(x) - 1)."""
    return jt.ternary(x >= 0, x, alpha * (jt.exp(x) - 1.0))


def _elu_derivative(x: jt.Var, alpha: float = 1.0) -> jt.Var:
    """d/dx ELU(x) = 1 if x >= 0 else alpha * exp(x)."""
    return jt.ternary(x >= 0, jt.ones_like(x), alpha * jt.exp(x))


def _get_activation(name: str):
    """Activation factory. Returns (forward_fn, derivative_fn)."""
    name = name.lower()
    if name == "elu":
        return _elu, _elu_derivative
    elif name == "swish":
        return _swish, _swish_derivative
    else:
        raise ValueError(f"Unknown activation_fn: {name!r}. Supported: elu, swish.")


class FCNet(nn.Module):

    def __init__(
        self,
        channel: int,
        preact: bool = False,
        nhidden: int = 2,
        idim: int = 64,
        coeff: float = 0.9,
        n_iterations: int | None = None,
        sn_atol: float = 1e-3,
        sn_rtol: float = 1e-3,
        activation_fn: str = "elu",
    ):
        super().__init__()
        self.preact = preact
        # 显式存 activation_fn, 让 Test 1b 能验证
        self.activation_fn = activation_fn
        self._act_fwd, self._act_deriv = _get_activation(activation_fn)

        layers = []
        last_dim_in = channel
        self._has_preact = bool(preact)

        for i in range(nhidden):
            layers.append(InducedNormLinear(
                last_dim_in, idim, bias=True,
                coeff=coeff, n_iterations=n_iterations,
                atol=sn_atol, rtol=sn_rtol,
            ))
            last_dim_in = idim

        layers.append(InducedNormLinear(
            last_dim_in, channel, bias=True,
            coeff=coeff, n_iterations=n_iterations,
            atol=sn_atol, rtol=sn_rtol,
        ))

        self.layers = nn.ModuleList(layers)

    def execute(self, x: jt.Var) -> jt.Var:
        h = x
        if self._has_preact:
            h = self._act_fwd(h)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self._act_fwd(h)
        return h

    def build_clone(self):
        cloned_layers = [layer.build_clone() for layer in self.layers]
        act_fwd = self._act_fwd

        class _FCNetClone(nn.Module):
            def __init__(self, layers_list, has_preact):
                super().__init__()
                self.layers = nn.ModuleList(layers_list)
                self._has_preact = has_preact

            def execute(self, x):
                h = x
                if self._has_preact:
                    h = act_fwd(h)
                for i, layer in enumerate(self.layers):
                    h = layer(h)
                    if i < len(self.layers) - 1:
                        h = act_fwd(h)
                return h

        return _FCNetClone(cloned_layers, self._has_preact)

    def build_jvp_net(self, x: jt.Var):
        # Use .detach() not .stop_grad() to avoid modifying input Var's flag.
        # See spectral_norm.build_clone for details on this Jittor pitfall.
        act_fwd = self._act_fwd
        act_deriv = self._act_deriv

        jvp_layers = []
        h = x.detach()
        if self._has_preact:
            h = act_fwd(h)
        for i, layer in enumerate(self.layers):
            m, h_out = layer.build_jvp_net(h)
            jvp_layers.append(m)
            if i < len(self.layers) - 1:
                h = act_fwd(h_out)
            else:
                h = h_out
        y = h

        class _FCNetJVP(nn.Module):
            def __init__(self, layers_list, has_preact, activations):
                super().__init__()
                self.layers = nn.ModuleList(layers_list)
                self._has_preact = has_preact
                self._activations = activations

            def execute(self, v):
                h = v
                if self._has_preact:
                    h = h * self._activations[0]
                for i, layer in enumerate(self.layers):
                    h = layer(h)
                    if i < len(self.layers) - 1:
                        h = h * self._activations[i + (1 if self._has_preact else 0)]
                return h

        # 用 act_deriv 算 JVP 的对角缩放
        activations = []
        h_track = x.detach()
        if self._has_preact:
            activations.append(act_deriv(h_track).detach())
            h_track = act_fwd(h_track)
        for i, layer in enumerate(self.layers):
            with jt.no_grad():
                h_track = layer.compute_weight(update=False) @ h_track.transpose(-1, -2)
                h_track = h_track.transpose(-1, -2)
                if layer.bias is not None:
                    h_track = h_track + layer.bias
            if i < len(self.layers) - 1:
                activations.append(act_deriv(h_track).detach())
                h_track = act_fwd(h_track)

        return _FCNetJVP(jvp_layers, self._has_preact, activations), y
