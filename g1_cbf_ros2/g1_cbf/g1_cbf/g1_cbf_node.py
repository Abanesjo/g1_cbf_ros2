#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid self-collision avoidance.

Subscribes to /joint_commands_unsafe, applies CBF-QP filtering to prevent
self-collisions between torso and arms, publishes safe commands on /joint_commands
at a fixed rate (1/dt Hz) regardless of upstream publish rate.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g1_cbf.scaling import Capsule3D
from g1_cbf.cbf import CapsuleCBF
from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS, COLLISION_PAIRS
from g1_cbf.qp_solver import CBFQPSolver
from g1_cbf.collider_viz import ColliderVisualizer


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # Parameters
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('beta', 1.05)
        self.declare_parameter('gamma', 5.0)
        self.declare_parameter('K', 5.0)
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('lpf_gain', 0.1)
        self.declare_parameter('urdf_path', '')

        dt = self.get_parameter('dt').value
        beta = self.get_parameter('beta').value
        gamma = self.get_parameter('gamma').value
        urdf_path = self.get_parameter('urdf_path').value

        if not urdf_path:
            self.get_logger().fatal('urdf_path parameter is required')
            raise RuntimeError('urdf_path not set')

        self.get_logger().info(f'Loading URDF: {urdf_path}')
        self.get_logger().info(f'CBF params: dt={dt}, beta={beta}, gamma={gamma}')

        # Subsystems
        self.kin = G1Kinematics(urdf_path)
        self.cbf = CapsuleCBF(beta=beta, gamma=gamma)
        self.qp = CBFQPSolver(n_joints=self.kin.n_q, n_cbf=len(COLLISION_PAIRS))
        self.viz = ColliderVisualizer(self, self.kin, beta)

        # Collision bodies (persistent, updated each tick)
        self.bodies = {}
        for name, body in self.kin.collision_bodies.items():
            self.bodies[name] = Capsule3D(
                np.zeros(3), np.eye(3),
                body['half_length'],
                body['radius'],
            )

        # State
        self.q_full = None  # Latest full joint state
        self.q_des_latest = None  # No command until upstream sends
        self.q_des_filtered = None
        self.q_cbf_target = None  # Integrated CBF-safe position
        self.nu_warm = {
            pair: 0.5 for pair in COLLISION_PAIRS
        }
        self.p_warm = {
            pair: None for pair in COLLISION_PAIRS
        }

        # Velocity limits (rad/s) from URDF
        self.dq_max = np.array([
            30.0, 30.0,
            37.0, 37.0, 37.0, 37.0,
            37.0, 37.0, 37.0, 37.0,
        ])
        self.dq_min = -self.dq_max

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10)
        self.create_subscription(JointState, '/joint_commands_unsafe', self._unsafe_cmd_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Timer: run CBF loop at 1/dt Hz
        self.create_timer(dt, self._tick)

        self.get_logger().info(f'g1_cbf_node ready — publishing at {1.0/dt:.0f} Hz')

    def _joint_states_cb(self, msg: JointState):
        """Store latest full joint state and publish collider visualization."""
        self.q_full = self.kin.joint_names_to_q_full(list(msg.name), list(msg.position))

    def _unsafe_cmd_cb(self, msg: JointState):
        """Store latest desired positions (do not process here — timer handles it)."""
        q_des = self._extract_controlled(msg)
        if q_des is not None:
            self.q_des_latest = q_des

    def _tick(self):
        """Timer callback: run CBF-QP and publish."""
        if self.q_full is None or self.q_des_latest is None:
            return

        dt = self.get_parameter('dt').value
        K = self.get_parameter('K').value
        max_vel = self.get_parameter('max_velocity').value
        lpf = self.get_parameter('lpf_gain').value
        q_ctrl = self.kin.extract_controlled(self.q_full)

        # Initialize targets on first tick
        if self.q_des_filtered is None:
            self.q_des_filtered = self.q_des_latest.copy()
        if self.q_cbf_target is None:
            self.q_cbf_target = q_ctrl.copy()
        if 0 < lpf < 1:
            self.q_des_filtered += lpf * (
                self.q_des_latest - self.q_des_filtered
            )
        else:
            self.q_des_filtered = self.q_des_latest.copy()

        # Proportional gain + velocity clamp
        dq_ref = K * (self.q_des_filtered - self.q_cbf_target)
        dq_ref = np.clip(dq_ref, -max_vel, max_vel)

        # Update FK
        self.kin.update(self.q_full)

        # Update body poses and get Jacobians
        jacobians = {}
        for name in self.bodies:
            center, R = self.kin.get_collision_pose(name)
            self.bodies[name].update(center, R)
            jacobians[name] = (
                self.kin.get_collision_jacobian(name)
            )

        # Publish collider visualization
        self.viz.publish(self.get_clock().now().to_msg())

        # Build CBF constraints
        constraints = []
        alpha_min = float('inf')
        for pair in COLLISION_PAIRS:
            nameA, nameB = pair
            alpha, A_row, b_val, nu_new, p_new = (
                self.cbf.build_constraint(
                    self.bodies[nameA],
                    self.bodies[nameB],
                    jacobians[nameA],
                    jacobians[nameB],
                    self.nu_warm[pair],
                    self.p_warm[pair],
                )
            )
            self.nu_warm[pair] = nu_new
            self.p_warm[pair] = p_new
            constraints.append((A_row, b_val))
            alpha_min = min(alpha_min, alpha)

        # Solve QP
        dq_safe = self.qp.solve(
            dq_ref, constraints,
            self.dq_min, self.dq_max,
        )

        # Integrate safe velocity into persistent target
        self.q_cbf_target += dq_safe * dt

        # Clamp to prevent divergence if blocked
        max_lead = 0.5
        self.q_cbf_target = np.clip(
            self.q_cbf_target,
            q_ctrl - max_lead,
            q_ctrl + max_lead,
        )

        safe_msg = JointState()
        stamp = self.get_clock().now().to_msg()
        safe_msg.header.stamp = stamp
        safe_msg.name = list(CONTROLLED_JOINTS)
        safe_msg.position = self.q_cbf_target.tolist()
        safe_msg.velocity = dq_safe.tolist()
        self.cmd_pub.publish(safe_msg)

        beta = self.get_parameter('beta').value
        if alpha_min < 3.0 * beta:
            self.get_logger().info(
                f'alpha_min={alpha_min:.4f} '
                f'beta={beta} '
                f'dq_ref={np.linalg.norm(dq_ref):.3f} '
                f'dq_safe={np.linalg.norm(dq_safe):.3f}',
                throttle_duration_sec=0.2,
            )

    def _extract_controlled(self, msg: JointState) -> np.ndarray:
        """Extract controlled joint positions from JointState message."""
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(self.kin.n_q)
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname not in name_to_pos:
                self.get_logger().warn(
                    f'Joint {jname} not in /joint_commands_unsafe, dropping',
                    throttle_duration_sec=2.0,
                )
                return None
            q[i] = name_to_pos[jname]
        return q


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
