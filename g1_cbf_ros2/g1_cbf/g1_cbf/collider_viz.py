"""MarkerArray publisher for collision capsule visualization.

Publishes /robot_colliders with all bodies as capsules
(cylinder + 2 spheres) at scale 1.0 — exactly matching
the CBF geometry.
"""

from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

_COLORS = {
    'torso': (0.2, 0.8, 0.2, 0.3),
    'left_arm': (0.2, 0.4, 0.9, 0.3),
    'right_arm': (0.9, 0.3, 0.2, 0.3),
}


class ColliderVisualizer:
    """Publishes MarkerArray for collision capsules."""

    def __init__(self, node, kinematics, beta: float = 1.05):
        self.kin = kinematics
        self.pub = node.create_publisher(
            MarkerArray, '/robot_colliders', 10,
        )

    def publish(self, stamp):
        """Publish capsule markers using current FK state."""
        msg = MarkerArray()
        mid = 0

        for name, body in self.kin.collision_bodies.items():
            radius = body['radius']
            half_len = body['half_length']
            seg_half = half_len - radius
            shaft_len = 2.0 * seg_half
            color = _COLORS.get(name, (0.5, 0.5, 0.5, 0.3))

            center, rot = self.kin.get_collision_pose(name)
            quat = Rot.from_matrix(rot).as_quat()
            axis = rot[:, 2]  # capsule long axis

            # Cylinder shaft
            diam = 2.0 * radius
            m = self._make_marker(
                stamp, mid, Marker.CYLINDER,
                center, quat,
                diam, diam, shaft_len,
                color,
            )
            msg.markers.append(m)
            mid += 1

            # Hemisphere spheres at +/- ends
            for sign in (+1, -1):
                sph_c = center + sign * seg_half * axis
                m = self._make_marker(
                    stamp, mid, Marker.SPHERE,
                    sph_c, quat,
                    diam, diam, diam,
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
