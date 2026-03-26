"""Capsule CBF using dpax for differentiable proximity computation.

Uses dpax.endpoints.proximity which returns
phi = d_centerline^2 - (R1+R2)^2.
Gradients via jax.grad with custom JVP.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from dpax.endpoints import proximity
from dpax.qp_utils import get_cost_terms, active_set_qp


class DpaxCBF:
    """CBF for capsule collision pairs using dpax."""

    def __init__(self, gamma: float = 5.0, margin_phi: float = 0.001):
        self.gamma = gamma
        self.margin_phi = margin_phi

        # JIT-compile the gradient function once
        self._grad_fn = jax.jit(
            grad(proximity, argnums=(1, 2, 4, 5))
        )

        # Warm up JAX JIT with dummy call
        _R = 0.1
        _a = jnp.zeros(3)
        _b = jnp.ones(3)
        _ = proximity(_R, _a, _b, _R, _a + 5.0, _b + 5.0)
        _ = self._grad_fn(_R, _a, _b, _R, _a + 5.0, _b + 5.0)

    def compute_phi_and_grad(self, R1, a1, b1, R2, a2, b2):
        """Compute proximity phi and endpoint gradients.

        Returns (phi, dphi_da1, dphi_db1, dphi_da2, dphi_db2).
        All returned as numpy arrays.
        """
        # Convert to JAX arrays (float64)
        R1_j = jnp.float64(R1)
        R2_j = jnp.float64(R2)
        a1_j = jnp.array(a1, dtype=jnp.float64)
        b1_j = jnp.array(b1, dtype=jnp.float64)
        a2_j = jnp.array(a2, dtype=jnp.float64)
        b2_j = jnp.array(b2, dtype=jnp.float64)

        phi = float(proximity(R1_j, a1_j, b1_j, R2_j, a2_j, b2_j))

        ga1, gb1, ga2, gb2 = self._grad_fn(
            R1_j, a1_j, b1_j, R2_j, a2_j, b2_j,
        )

        return (
            phi,
            np.asarray(ga1),
            np.asarray(gb1),
            np.asarray(ga2),
            np.asarray(gb2),
        )

    def build_constraint(
        self,
        R1, a1, b1, J_a1, J_b1,
        R2, a2, b2, J_a2, J_b2,
    ):
        """Build one CBF constraint for a capsule pair.

        Returns (phi, A_row, b_val) where the constraint is:
            A_row @ dq >= b_val
        i.e. dphi/dq @ dq >= -gamma * (phi - margin_phi)
        """
        phi, ga1, gb1, ga2, gb2 = self.compute_phi_and_grad(
            R1, a1, b1, R2, a2, b2,
        )

        # Chain rule: dphi/dq = sum of dphi/d(endpoint) @ J_endpoint
        dphi_dq = (
            ga1 @ J_a1 + gb1 @ J_b1
            + ga2 @ J_a2 + gb2 @ J_b2
        )

        h = phi - self.margin_phi
        A_row = dphi_dq
        b_val = -self.gamma * h

        # Recover closest points on centerlines via dpax QP
        a1_j = jnp.array(a1, dtype=jnp.float64)
        b1_j = jnp.array(b1, dtype=jnp.float64)
        a2_j = jnp.array(a2, dtype=jnp.float64)
        b2_j = jnp.array(b2, dtype=jnp.float64)
        Q, q, _ = get_cost_terms(a1_j, b1_j, a2_j, b2_j)
        z = active_set_qp(Q, q)
        p1 = np.asarray(b1_j + z[0] * (a1_j - b1_j))
        p2 = np.asarray(b2_j + z[1] * (a2_j - b2_j))

        return phi, A_row, b_val, p1, p2
