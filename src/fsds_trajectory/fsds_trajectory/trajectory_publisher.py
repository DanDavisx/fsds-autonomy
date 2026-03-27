#!/usr/bin/env python3
"""
Trajectory Publisher

This node publishes a reference driving path for the FSDS.
The track layout can be described by a cone map CSV. This CSV has blue cones on one boundary,
yellow cones on the opposite, and big orange cones signifying the starting line.
This node computes a centreline path from the cone geometry and uses it as the reference trajectory
for the MPC.

You can call this function using:
ros2 run fsds_trajectory trajectory_publisher --ros-args -p csv_path:=/yourcsvpath -p frame_id:=fsds/map
"""

import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cross2(ax, ay, bx, by):
    return ax * by - ay * bx


class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')

        # - PARAMETERS -
        self.declare_parameter('csv_path', '')
        self.declare_parameter('frame_id', 'fsds/map')
        self.declare_parameter('topic', '/reference_path')
        self.declare_parameter('blue_boundary_topic', '/track_boundary_blue')
        self.declare_parameter('yellow_boundary_topic', '/track_boundary_yellow')
        self.declare_parameter('blue_cones_topic', '/track_blue_cones')
        self.declare_parameter('yellow_cones_topic', '/track_yellow_cones')
        self.declare_parameter('orange_cones_topic', '/track_orange_cones')
        self.declare_parameter('ds', 0.5)
        self.declare_parameter('smooth_iterations', 3)

        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        topic = self.get_parameter('topic').value
        blue_boundary_topic = self.get_parameter('blue_boundary_topic').value
        yellow_boundary_topic = self.get_parameter('yellow_boundary_topic').value
        blue_cones_topic = self.get_parameter('blue_cones_topic').value
        yellow_cones_topic = self.get_parameter('yellow_cones_topic').value
        orange_cones_topic = self.get_parameter('orange_cones_topic').value
        self.ds = float(self.get_parameter('ds').value)
        smooth_iterations = int(self.get_parameter('smooth_iterations').value)

        # - PUBLISHERS -
        from rclpy.qos import QoSProfile, DurabilityPolicy
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub = self.create_publisher(NavPath, topic, qos)
        self.blue_boundary_pub = self.create_publisher(NavPath, blue_boundary_topic, qos)
        self.yellow_boundary_pub = self.create_publisher(NavPath, yellow_boundary_topic, qos)
        self.blue_cones_pub = self.create_publisher(MarkerArray, blue_cones_topic, qos)
        self.yellow_cones_pub = self.create_publisher(MarkerArray, yellow_cones_topic, qos)
        self.orange_cones_pub = self.create_publisher(MarkerArray, orange_cones_topic, qos)

        # - LOADING CONES FROM CSV -
        if not csv_path:
            self.get_logger().error('No CSV path provided. Add --ros-args -p csv_path:=/path/to/track.csv')
            return

        self.cones_blue, self.cones_yellow, self.cones_orange = self.load_cones(Path(csv_path))
        self.get_logger().info(
            f'Successfully loaded cones: blue={len(self.cones_blue)}, yellow={len(self.cones_yellow)}, big_orange={len(self.cones_orange)}'
        )

        start_hint = self.start_line_midpoint(self.cones_orange, self.cones_blue, self.cones_yellow)

        self.blue_boundary = self.close_loop(self.order_boundary_loop(self.cones_blue, start_hint))
        self.yellow_boundary = self.close_loop(self.order_boundary_loop(self.cones_yellow, start_hint))

        # - PATH CALCULATION -
        centreline_seed = self.build_centreline_from_boundaries(
            self.blue_boundary,
            self.yellow_boundary,
            self.cones_orange,
            samples=max(200, 3 * max(len(self.cones_blue), len(self.cones_yellow))),
        )

        if len(centreline_seed) < 4:
            self.get_logger().error('Failed to build a valid centreline from the cone map.')
            return

        self.get_logger().info(f'Successfully built centreline with {len(centreline_seed)} points')

        # - PATH REFINEMENT -
        centreline_closed = self.close_loop(centreline_seed)
        centreline_smooth = self.smooth_path(centreline_closed, iterations=smooth_iterations)
        centreline_final = self.resample_by_distance(centreline_smooth, ds=self.ds)
        centreline_final = self.ensure_blue_on_left(centreline_final, self.blue_boundary)
        centreline_final = self.rotate_closed_path_to_start(centreline_final, start_hint)

        # - MESSAGE BUILD -
        self.path_msg = self.build_path(centreline_final)
        self.blue_boundary_msg = self.build_path(self.blue_boundary)
        self.yellow_boundary_msg = self.build_path(self.yellow_boundary)
        self.blue_markers = self.build_cone_markers(self.cones_blue, 'blue')
        self.yellow_markers = self.build_cone_markers(self.cones_yellow, 'yellow')
        self.orange_markers = self.build_cone_markers(self.cones_orange, 'orange')

        self.publish_all()
        self.timer = self.create_timer(2.0, self.timer_cb)

    def timer_cb(self):
        self.publish_all()

    def publish_all(self):
        stamp = self.get_clock().now().to_msg()

        self.path_msg.header.stamp = stamp
        for ps in self.path_msg.poses:
            ps.header.stamp = stamp
        self.pub.publish(self.path_msg)

        self.blue_boundary_msg.header.stamp = stamp
        for ps in self.blue_boundary_msg.poses:
            ps.header.stamp = stamp
        self.blue_boundary_pub.publish(self.blue_boundary_msg)

        self.yellow_boundary_msg.header.stamp = stamp
        for ps in self.yellow_boundary_msg.poses:
            ps.header.stamp = stamp
        self.yellow_boundary_pub.publish(self.yellow_boundary_msg)

        self._stamp_markers(self.blue_markers, stamp)
        self._stamp_markers(self.yellow_markers, stamp)
        self._stamp_markers(self.orange_markers, stamp)
        self.blue_cones_pub.publish(self.blue_markers)
        self.yellow_cones_pub.publish(self.yellow_markers)
        self.orange_cones_pub.publish(self.orange_markers)

    def _stamp_markers(self, marker_array, stamp):
        for marker in marker_array.markers:
            marker.header.stamp = stamp

    def load_cones(self, csv_path: Path):
        blue, yellow, orange = [], [], []
        with csv_path.open('r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                tag = row[0].strip()
                x = float(row[1])
                y = float(row[2])
                if tag == 'blue':
                    blue.append((x, y))
                elif tag == 'yellow':
                    yellow.append((x, y))
                elif tag == 'big_orange':
                    orange.append((x, y))
        return blue, yellow, orange

    # - CENTRELINE CONSTRUCTOR -
    def build_centreline_from_boundaries(self, blue, yellow, orange, samples=200):
        if len(blue) < 3 or len(yellow) < 3:
            return []

        blue_samples = self.resample_closed_n(blue, samples)
        yellow_samples_fwd = self.resample_closed_n(yellow, samples)
        yellow_samples_rev = self.resample_closed_n(list(reversed(yellow[:-1] if yellow and yellow[0] == yellow[-1] else yellow)), samples)

        fwd_score, fwd_shift = self.best_cyclic_alignment(blue_samples, yellow_samples_fwd)
        rev_score, rev_shift = self.best_cyclic_alignment(blue_samples, yellow_samples_rev)

        if rev_score < fwd_score:
            yellow_samples = self.rotate_list(yellow_samples_rev, rev_shift)
            chosen = 'reversed'
            chosen_score = rev_score
        else:
            yellow_samples = self.rotate_list(yellow_samples_fwd, fwd_shift)
            chosen = 'forward'
            chosen_score = fwd_score

        centreline = [
            ((bx + yx) * 0.5, (by + yy) * 0.5)
            for (bx, by), (yx, yy) in zip(blue_samples, yellow_samples)
        ]

        centreline = self.close_loop(centreline)
        centreline = self.ensure_blue_on_left(centreline, blue)
        centreline = self.rotate_closed_path_to_start(centreline, self.start_line_midpoint(orange, blue, yellow))
        return centreline

    # - STARTING LINE (ORANGE CONES) -
    def start_line_midpoint(self, orange, blue, yellow):
        if len(orange) >= 2:
            return (
                0.5 * (orange[0][0] + orange[1][0]),
                0.5 * (orange[0][1] + orange[1][1]),
            )

        all_pts = list(blue) + list(yellow)
        if not all_pts:
            return (0.0, 0.0)

        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)
        return (cx, cy)

    def order_boundary_loop(self, points, start_hint):
        if len(points) < 3:
            return list(points)

        pts = list(points)
        start_idx = min(range(len(pts)), key=lambda i: dist2(pts[i], start_hint))
        ordered = [pts.pop(start_idx)]

        second_idx = min(range(len(pts)), key=lambda i: dist2(pts[i], ordered[0]))
        ordered.append(pts.pop(second_idx))

        while pts:
            prev = ordered[-2]
            curr = ordered[-1]
            step = (curr[0] - prev[0], curr[1] - prev[1])
            step_norm = math.hypot(step[0], step[1])

            def candidate_cost(p):
                v = (p[0] - curr[0], p[1] - curr[1])
                d = math.hypot(v[0], v[1])
                if d < 1e-12:
                    return float('inf')

                if step_norm < 1e-12:
                    turn_penalty = 0.0
                else:
                    cosang = (step[0] * v[0] + step[1] * v[1]) / (step_norm * d)
                    cosang = max(-1.0, min(1.0, cosang))
                    turn_penalty = 1.0 - cosang

                return d + 2.5 * turn_penalty

            next_idx = min(range(len(pts)), key=lambda i: candidate_cost(pts[i]))
            ordered.append(pts.pop(next_idx))

        return ordered

    def resample_closed_n(self, pts, n):
        pts = self.close_loop(list(pts))
        if len(pts) < 2 or n <= 0:
            return list(pts)

        seg_lengths = [dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        total_len = sum(seg_lengths)
        if total_len < 1e-12:
            return [pts[0]] * n

        samples = []
        target_step = total_len / float(n)
        seg_idx = 0
        seg_start_s = 0.0

        for k in range(n):
            s = k * target_step
            while seg_idx < len(seg_lengths) - 1 and seg_start_s + seg_lengths[seg_idx] < s:
                seg_start_s += seg_lengths[seg_idx]
                seg_idx += 1

            p0 = pts[seg_idx]
            p1 = pts[seg_idx + 1]
            seg_len = seg_lengths[seg_idx]
            if seg_len < 1e-12:
                samples.append(p0)
                continue

            t = (s - seg_start_s) / seg_len
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            samples.append((x, y))

        return samples

    def best_cyclic_alignment(self, reference, candidate):
        n = min(len(reference), len(candidate))
        if n == 0:
            return float('inf'), 0

        best_score = float('inf')
        best_shift = 0
        for shift in range(n):
            score = 0.0
            for i in range(n):
                score += dist(reference[i], candidate[(i + shift) % n])
            score /= float(n)
            if score < best_score:
                best_score = score
                best_shift = shift

        return best_score, best_shift

    def rotate_list(self, pts, shift):
        if not pts:
            return []
        shift %= len(pts)
        return pts[shift:] + pts[:shift]

    def rotate_closed_path_to_start(self, pts, start_hint):
        if not pts:
            return pts

        pts = list(pts)
        closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6
        core = pts[:-1] if closed else pts
        if not core:
            return pts

        start_idx = min(range(len(core)), key=lambda i: dist2(core[i], start_hint))
        rotated = core[start_idx:] + core[:start_idx]
        return self.close_loop(rotated)

    def ensure_blue_on_left(self, centreline, blue_boundary):
        pts = self.close_loop(list(centreline))
        blue_pts = list(blue_boundary[:-1]) if len(blue_boundary) > 1 and blue_boundary[0] == blue_boundary[-1] else list(blue_boundary)
        if len(pts) < 3 or len(blue_pts) < 1:
            return pts

        total_cross = 0.0
        core = pts[:-1]
        for p0, p1 in zip(core, core[1:] + core[:1]):
            mx, my = p0
            tx = p1[0] - mx
            ty = p1[1] - my
            bi = min(range(len(blue_pts)), key=lambda i: dist2(blue_pts[i], p0))
            bx, by = blue_pts[bi]
            total_cross += cross2(tx, ty, bx - mx, by - my)

        if total_cross < 0.0:
            return self.close_loop(list(reversed(core)))
        return pts

    # - PATH UTILITIES - 

    # - CLOSE LOOP HELPER FUNCTION -
    # Close loop helper. Simly ensures last point equals first.
    def close_loop(self, pts, eps=1e-6):
        if not pts:
            return pts
        first = pts[0]
        last = pts[-1]

        # Compare distance between first and last
        if math.hypot(first[0] - last[0], first[1] - last[1]) > eps:
            pts = list(pts) + [first]
        return pts

    # - PATH SMOOTHING -
    # This essentially slices every corner of a polygon to get a smoother curve.
    # Improves MPC accuracy by fixing jagged lines in the trajectory path.
    # Chaikin's closed loop algorithm.
    def smooth_path(self, pts, iterations=3):
        if len(pts) < 4:
            return pts

        # Ensure closed loop.
        pts = self.close_loop(list(pts))
        for _ in range(iterations):
            new_pts = []
            # Work through each segment.
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i + 1]

                # Generate 2 new points for each segment, Q and R.
                qx = 0.75 * p0[0] + 0.25 * p1[0]
                qy = 0.75 * p0[1] + 0.25 * p1[1]
                rx = 0.25 * p0[0] + 0.75 * p1[0]
                ry = 0.25 * p0[1] + 0.75 * p1[1]

                # Add these new points.
                new_pts.append((qx, qy))
                new_pts.append((rx, ry))
            
            # Close the loop again.
            new_pts.append(new_pts[0])
            pts = new_pts
        return pts

    # - CHECKPOINT DISTANCE RESAMPLE FUNCTION -
    # Resampling to space cone checkpoints evenly.
    # Once again helps out with MPC accuracy.
    def resample_by_distance(self, pts, ds=0.5):
        if len(pts) < 2:
            return pts

        pts = list(pts)

        # Guarantee closed loop.
        closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-9
        if not closed:
            pts.append(pts[0])

        out = [pts[0]] # New resampled list.
        acc = 0.0 # How much distance gets carried over from previous segments.

        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            seg_dx = x1 - x0
            seg_dy = y1 - y0
            seg_len = math.hypot(seg_dx, seg_dy) # Length of current segment. 
            if seg_len < 1e-12:
                continue

            dist_along = 0.0
            while acc + (seg_len - dist_along) >= ds:
                remaining = ds - acc
                t = (dist_along + remaining) / seg_len
                xn = x0 + t * seg_dx
                yn = y0 + t * seg_dy
                out.append((xn, yn))
                dist_along += remaining
                acc = 0.0

            acc += (seg_len - dist_along)
        
        # Close the resampled path.
        if out and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) > 1e-6:
            out.append(out[0])

        return out

    # - BUILD ROS MESSAGE PATH FUNCTION -
    # Converts the final list of resampled x,y points into a nav_msgs/Path message. 
    def build_path(self, points_xy):
        msg = NavPath()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        for (x, y) in points_xy:
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.header.stamp = msg.header.stamp
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        return msg
    
    def build_cone_markers(self, cones, colour_name):
        msg = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        colour_map = {
            'blue': (0.0, 0.2, 1.0),
            'yellow': (1.0, 1.0, 0.0),
            'orange': (1.0, 0.5, 0.0),
        }
        r, g, b = colour_map[colour_name]
        for i, (x, y) in enumerate(cones):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.ns = colour_name + '_cones'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.35
            marker.color.a = 1.0
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            msg.markers.append(marker)
        return msg


def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()