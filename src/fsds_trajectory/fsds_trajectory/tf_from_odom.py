#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class TfFromOdom(Node):
    """
    Subscribes to an Odometry topic and republishes the pose as a TF transform.

    This is needed when a system provides Odometry messages with frame_id/child_frame_id
    but does not publish the corresponding dynamic TF on /tf. RViz requires TF frames
    to exist in order to set a Fixed Frame and transform visualized data.
    """

    def __init__(self):
        super().__init__('tf_from_odom')

        self.declare_parameter('odom_topic', '/testing_only/odom')
        odom_topic = self.get_parameter('odom_topic').value

        self._br = TransformBroadcaster(self)

        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.get_logger().info(f"Relaying TF from Odometry on: {odom_topic}")

    def odom_cb(self, msg: Odometry):
        # Build TF: parent = odom.header.frame_id, child = odom.child_frame_id
        t = TransformStamped()
        t.header.stamp = msg.header.stamp  # use sim time stamp
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = msg.child_frame_id

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self._br.sendTransform(t)


def main():
    rclpy.init()
    node = TfFromOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
