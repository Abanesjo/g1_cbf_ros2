#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid self-collision avoidance.

Subscribes to /joint_commands_unsafe, applies CBF-QP filtering to prevent
self-collisions between torso and arms, publishes safe commands on
/joint_commands at a fixed rate (1/dt Hz).

Supports capsule and box collision geometry via collision_geometry param.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g1_cbf.cbf import DpaxCapsuleCBF, DpaxBoxCBF
from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS, COLLISION_PAIRS
from g1_cbf.qp_solver import CBFQPSolver
from g1_cbf.collider_viz import ColliderVisualizer


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # Parameters
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('margin_phi', 0.001)
        self.declare_parameter('beta', 1.05)
        self.declare_parameter('gamma', 5.0)
        self.declare_parameter('K', 5.0)
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('lpf_gain', 0.1)
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('collision_geometry', 'capsules')

        dt = self.get_parameter('dt').value
        gamma = self.get_parameter('gamma').value
        urdf_path = self.get_parameter('urdf_path').value
        self.geom_type = self.get_parameter('collision_geometry').value

        if not urdf_path:
            self.get_logger().fatal('urdf_path parameter is required')
            raise RuntimeError('urdf_path not set')

        self.get_logger().info(f'Loading URDF: {urdf_path}')
        self.get_logger().info(
            f'CBF params: dt={dt}, gamma={gamma}, '
            f'geometry={self.geom_type}'
        )

        # Subsystems
        self.kin = G1Kinematics(urdf_path)
        self.get_logger().info('Initializing dpax CBF (JAX JIT warmup)...')

        if self.geom_type == 'boxes':
            beta = self.get_parameter('beta').value
            self.cbf = DpaxBoxCBF(gamma=gamma, beta=beta)
        else:
            margin_phi = self.get_parameter('margin_phi').value
            self.cbf = DpaxCapsuleCBF(
                gamma=gamma, margin_phi=margin_phi,
            )

        self.get_logger().info('dpax CBF ready')
        self.qp = CBFQPSolver(
            n_joints=self.kin.n_q,
            n_cbf=len(COLLISION_PAIRS),
        )
        self.viz = ColliderVisualizer(
            self, self.kin, geometry_type=self.geom_type,
        )

        # State
        self.q_full = None
        self.q_des_latest = None
        self.q_des_filtered = None
        self.q_cbf_target = None

        # Velocity limits (rad/s)
        self.dq_max = np.array([
            30.0, 30.0,
            37.0, 37.0, 37.0, 37.0,
            37.0, 37.0, 37.0, 37.0,
        ])
        self.dq_min = -self.dq_max

        # Subscribers
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_states_cb, 10,
        )
        self.create_subscription(
            JointState, '/joint_commands_unsafe',
            self._unsafe_cmd_cb, 10,
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10,
        )

        # Timer
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'g1_cbf_node ready — publishing at {1.0/dt:.0f} Hz'
        )

    def _joint_states_cb(self, msg: JointState):
        self.q_full = self.kin.joint_names_to_q_full(
            list(msg.name), list(msg.position),
        )

    def _unsafe_cmd_cb(self, msg: JointState):
        q_des = self._extract_controlled(msg)
        if q_des is not None:
            self.q_des_latest = q_des

    def _tick(self):
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

        # Publish collider visualization
        self.viz.publish(self.get_clock().now().to_msg())

        # Build CBF constraints
        constraints = []
        closest_points = []
        metric_min = float('inf')

        if self.geom_type == 'boxes':
            self._build_box_constraints(
                constraints, closest_points, metric_min,
            )
        else:
            self._build_capsule_constraints(
                constraints, closest_points, metric_min,
            )

        # Publish distance lines
        self.viz.publish_distances(
            self.get_clock().now().to_msg(), closest_points,
        )

        # Solve QP
        dq_safe = self.qp.solve(
            dq_ref, constraints,
            self.dq_min, self.dq_max,
        )

        # Integrate safe velocity into persistent target
        self.q_cbf_target += dq_safe * dt

        # Clamp to prevent divergence
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

    def _build_capsule_constraints(self, constraints, closest_points,
                                   metric_min):
        endpoints = {}
        for name in self.kin.collision_bodies:
            a, b, J_a, J_b = self.kin.get_endpoint_jacobians(name)
            body = self.kin.collision_bodies[name]
            endpoints[name] = {
                'a': a, 'b': b,
                'J_a': J_a, 'J_b': J_b,
                'radius': body['radius'],
            }

        for pair in COLLISION_PAIRS:
            nameA, nameB = pair
            eA, eB = endpoints[nameA], endpoints[nameB]

            phi, A_row, b_val, p1, p2 = self.cbf.build_constraint(
                eA['radius'], eA['a'], eA['b'],
                eA['J_a'], eA['J_b'],
                eB['radius'], eB['a'], eB['b'],
                eB['J_a'], eB['J_b'],
            )
            constraints.append((A_row, b_val))
            closest_points.append((p1, p2))

            margin_phi = self.get_parameter('margin_phi').value
            if phi < 3.0 * margin_phi:
                self.get_logger().info(
                    f'phi_min={phi:.6f} margin={margin_phi}',
                    throttle_duration_sec=0.2,
                )

    def _build_box_constraints(self, constraints, closest_points,
                               metric_min):
        for pair in COLLISION_PAIRS:
            nameA, nameB = pair
            bodyA = self.kin.collision_bodies[nameA]
            bodyB = self.kin.collision_bodies[nameB]
            centerA, rotA = self.kin.get_collision_pose(nameA)
            centerB, rotB = self.kin.get_collision_pose(nameB)
            J6_A = self.kin.get_collision_jacobian(nameA)
            J6_B = self.kin.get_collision_jacobian(nameB)

            alpha, A_row, b_val, p1, p2 = self.cbf.build_constraint(
                bodyA, centerA, rotA, J6_A,
                bodyB, centerB, rotB, J6_B,
            )
            constraints.append((A_row, b_val))
            closest_points.append((p1, p2))

            beta = self.get_parameter('beta').value
            if alpha < 1.5 * beta:
                self.get_logger().info(
                    f'alpha={alpha:.4f} beta={beta}',
                    throttle_duration_sec=0.2,
                )

    def _extract_controlled(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(self.kin.n_q)
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname not in name_to_pos:
                self.get_logger().warn(
                    f'Joint {jname} missing from '
                    f'/joint_commands_unsafe, dropping',
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
