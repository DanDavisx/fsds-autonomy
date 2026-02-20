#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class TfFromOdom(Node):
    """
    tf_from_odom.py
    
    The FSDS provides odometry messages with frame_id/child_frame_id but doesn't publish the dynamic TF on /tf.
    To fix this, the program subscribes to an odometry topic and republishes the pose as a TF transform.
    
    This program is needed to set fsds\map as a fixed frame in RVIZ.
    """

    def __init__(self):
        super().__init__('tf_from_odom')

        self.declare_parameter('odom_topic', '/testing_only/odom')
        odom_topic = self.get_parameter('odom_topic').value

        self._br = TransformBroadcaster(self)

        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.get_logger().info(f"Relaying TF from Odometry on: {odom_topic}")

    def odom_cb(self, msg: Odometry):
        # parent = odom.header.frame_id
        # child = odom.child_frame_id
        t = TransformStamped()
        t.header.stamp = msg.header.stamp  
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
