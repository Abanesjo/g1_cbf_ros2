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

from g1_cbf.scaling import Ellipsoid3D
from g1_cbf.cbf import EllipsoidCBF3D
from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS, COLLISION_PAIRS
from g1_cbf.qp_solver import CBFQPSolver
from g1_cbf.collider_viz import ColliderVisualizer


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # Parameters
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('beta', 1.05)
        self.declare_parameter('gamma', 5.0)
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
        self.cbf = EllipsoidCBF3D(beta=beta, gamma=gamma)
        self.qp = CBFQPSolver(n_joints=self.kin.n_q, n_cbf=len(COLLISION_PAIRS))
        self.viz = ColliderVisualizer(self, self.kin, beta)

        # Ellipsoid objects (persistent, updated each tick)
        self.ellipsoids = {}
        for name, body in self.kin.collision_bodies.items():
            self.ellipsoids[name] = Ellipsoid3D(
                np.zeros(3), np.eye(3), body['semi_axes']
            )

        # State
        self.q_full = None  # Latest full joint state from /joint_states
        # Default desired: all zeros (center all joints) until upstream sends something
        self.q_des_latest = np.zeros(self.kin.n_q)
        self.nu_warm = {pair: 0.5 for pair in COLLISION_PAIRS}

        # Velocity limits (rad/s) — from URDF joint limits
        # waist_roll=30, waist_pitch=30, L_sh_pitch=37, L_sh_roll=37, L_sh_yaw=37,
        # L_elbow=37, R_sh_pitch=37, R_sh_roll=37, R_sh_yaw=37, R_elbow=37
        self.dq_max = np.array([30.0, 30.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0])
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
        """Timer callback: run CBF-QP and publish safe command."""
        if self.q_full is None:
            return

        dt = self.get_parameter('dt').value

        # Current controlled positions
        q_ctrl = self.kin.extract_controlled(self.q_full)

        # Position -> velocity (interpolation toward latest desired)
        dq_ref = (self.q_des_latest - q_ctrl) / dt

        # Update FK with current full state
        self.kin.update(self.q_full)

        # Update ellipsoid poses and get Jacobians
        jacobians = {}
        for name in self.ellipsoids:
            center, R = self.kin.get_collision_pose(name)
            self.ellipsoids[name].update(center, R)
            jacobians[name] = self.kin.get_collision_jacobian(name)

        # Publish collider visualization
        self.viz.publish(self.get_clock().now().to_msg())

        # Build CBF constraints
        constraints = []
        alpha_min = float('inf')
        for pair in COLLISION_PAIRS:
            nameA, nameB = pair
            alpha, A_row, b_val, nu_new = self.cbf.build_constraint(
                self.ellipsoids[nameA], self.ellipsoids[nameB],
                jacobians[nameA], jacobians[nameB],
                self.nu_warm[pair],
            )
            self.nu_warm[pair] = nu_new
            constraints.append((A_row, b_val))
            alpha_min = min(alpha_min, alpha)

        # Solve QP
        dq_safe = self.qp.solve(dq_ref, constraints, self.dq_min, self.dq_max)

        # Convert back to position
        q_safe = q_ctrl + dq_safe * dt

        # Publish
        safe_msg = JointState()
        safe_msg.header.stamp = self.get_clock().now().to_msg()
        safe_msg.name = list(CONTROLLED_JOINTS)
        safe_msg.position = q_safe.tolist()
        self.cmd_pub.publish(safe_msg)

        self.get_logger().debug(
            f'alpha_min={alpha_min:.3f}, beta={self.cbf.beta}, '
            f'dq_ref_norm={np.linalg.norm(dq_ref):.3f}, '
            f'dq_safe_norm={np.linalg.norm(dq_safe):.3f}',
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
