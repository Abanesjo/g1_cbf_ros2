"""MarkerArray publisher for collision body visualization in RViz.

Publishes /robot_colliders:
- Torso: cube scaled by beta (linear scaling, paper eq 41)
- Arms: capsule (cylinder + 2 spheres), scaled by sqrt(beta)

Poses are computed from pinocchio FK (not TF).
"""

import math
from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

# Capsule params from URDF collision geometry
_ARM_RADIUS = 0.03
_ARM_LENGTH = 0.4
_ARM_HALF_LEN = _ARM_LENGTH / 2.0
# Cylinder shaft (straight portion between hemisphere centers)
_SHAFT_LEN = _ARM_LENGTH - 2 * _ARM_RADIUS  # 0.28
# Sphere center offset from capsule center along long axis
_SPHERE_OFFSET = _ARM_HALF_LEN - _ARM_RADIUS  # 0.14

_TORSO_SIZE = (0.16, 0.17, 0.525)

_COLORS = {
    'torso': (0.2, 0.8, 0.2, 0.3),
    'left_arm': (0.2, 0.4, 0.9, 0.3),
    'right_arm': (0.9, 0.3, 0.2, 0.3),
}


class ColliderVisualizer:
    """Publishes MarkerArray for collision bodies."""

    def __init__(self, node, kinematics, beta: float = 1.05):
        self.kin = kinematics
        self.beta = beta
        self.scale_factor = math.sqrt(beta)
        self.pub = node.create_publisher(
            MarkerArray, '/robot_colliders', 10,
        )

    def publish(self, stamp):
        """Publish markers using current FK state."""
        msg = MarkerArray()
        s = self.scale_factor
        mid = 0

        # --- Torso (cube, linear scaling by beta) ---
        b = self.beta
        center, rot = self.kin.get_collision_pose('torso')
        quat = Rot.from_matrix(rot).as_quat()
        m = self._make_marker(
            stamp, mid, Marker.CUBE, center, quat,
            _TORSO_SIZE[0] * b,
            _TORSO_SIZE[1] * b,
            _TORSO_SIZE[2] * b,
            _COLORS['torso'],
        )
        msg.markers.append(m)
        mid += 1

        # --- Arms (capsule = cylinder + 2 spheres each) ---
        for arm_name in ('left_arm', 'right_arm'):
            center, rot = self.kin.get_collision_pose(
                arm_name,
            )
            quat = Rot.from_matrix(rot).as_quat()
            color = _COLORS[arm_name]

            # Cylinder shaft
            diameter = 2 * _ARM_RADIUS * s
            m = self._make_marker(
                stamp, mid, Marker.CYLINDER,
                center, quat,
                diameter, diameter, _SHAFT_LEN * s,
                color,
            )
            msg.markers.append(m)
            mid += 1

            # Hemisphere spheres at +/- along capsule axis
            # Capsule long axis in world = rot's Z column
            axis_world = rot[:, 2]
            sphere_diam = 2 * _ARM_RADIUS * s

            for sign in (+1, -1):
                sph_center = (
                    center
                    + sign * _SPHERE_OFFSET * s
                    * axis_world
                )
                m = self._make_marker(
                    stamp, mid, Marker.SPHERE,
                    sph_center, quat,
                    sphere_diam, sphere_diam, sphere_diam,
                    color,
                )
                msg.markers.append(m)
                mid += 1

        self.pub.publish(msg)

    @staticmethod
    def _make_marker(
        stamp, marker_id, marker_type,
        center, quat, sx, sy, sz, color,
    ):
        m = Marker()
        m.header.frame_id = 'pelvis'
        m.header.stamp = stamp
        m.ns = 'colliders'
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position = Point(
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
        )
        m.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]),
            z=float(quat[2]), w=float(quat[3]),
        )
        m.scale = Vector3(
            x=float(sx), y=float(sy), z=float(sz),
        )
        r, g, b, a = color
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        return m
