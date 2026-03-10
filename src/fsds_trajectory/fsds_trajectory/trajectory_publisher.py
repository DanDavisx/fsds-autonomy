"""
Trajectory Publisher

This node publishes a reference driving path for the FSDS.
The track layout can be described by a cone map CSV. This CSV has blue cones on one boundary, yellow cones on the opposite, and big orange cones signifying the starting line.
This node will compute a centreline path from the cone geometry and use it as the reference trajectory for the MPC.
Please refer to FSDS documentation on how to format a cone map CSV.
The circuit has to be a closed loop circuit. 

You can call this function using:
ros2 run fsds_trajectory trajectory_publisher --ros-args -p csv_path:=/yourcsvpath -p frame_id:=fsds/map
In my case (for testing) my csv directory is /home/dan/Formula-Student-Driverless-Simulator/ros2/src/fsds_trajectory/maps/track_droneport.csv

Then, run rviz and set your frame to fsds/map, add a path, and set the path to /reference_path and also set the durability policy to transient local. You'll see the path appear. 
"""
#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped


# Finds squared distance between two 2d points.
# Uses squared distance to compare which cone or midpoint is nearest.
def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
 
        # - PARAMETERS - 
        self.declare_parameter('csv_path', '') # Where the cone CSV file is.
        self.declare_parameter('frame_id', 'fsds/map') # Coordinate frame for the path.
        self.declare_parameter('topic', '/reference_path') # Where to publish the reference path.

        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().Publisher_value if False else self.get_parameter('topic').value

        # - PUBLISHER - 
        from rclpy.qos import QoSProfile, DurabilityPolicy
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL 
        self.pub = self.create_publisher(NavPath, topic, qos)

        # - LOADING CONES FROM CSV -
        # No CSV Provided
        if not csv_path:
            self.get_logger().error("No CSV path provided. Add --ros-args -p csv_path:=/path/to/track.csv")
            return

        # Load cones and split by colour.
        self.cones_blue, self.cones_yellow, self.cones_orange = self.load_cones(Path(csv_path))
        self.get_logger().info(f"Loaded cones: blue={len(self.cones_blue)}, yellow={len(self.cones_yellow)}, big_orange={len(self.cones_orange)}")

        # - PATH CALCULATION -
        # Calculate the midpoints between the blue and yellow cones.
        midpoints = self.compute_midpoints_mutual_nn(self.cones_blue, self.cones_yellow)
        self.get_logger().info(f"Computed midpoint pairs: {len(midpoints)}")

        # Order the midpoints to produce the correct path.
        midpoints_ordered = self.order_points_nearest_neighbour(midpoints)
        # Reverse that list to allow for correct travel direction. 
        midpoints_ordered = list(reversed(midpoints_ordered))
        self.get_logger().info(f"Midpoints ordered: {midpoints_ordered}")
        # Close path so last point == first point which makes it closed loop circuit. 
        midpoints_closed = self.close_loop(midpoints_ordered)

        # - PATH REFINEMENT -
        # Apply path smoothing.
        midpoints_smooth = self.smooth_path(midpoints_closed, iterations=3)

        # Resample to uniform spacing. Cone checkpoints are 0.5 m apart.
        midpoints_final = self.resample_by_distance(midpoints_smooth, ds=0.5)

        # - PATH BUILD & PUBLISH- 
        # Build and publish the path.
        self.path_msg = self.build_path(midpoints_final)
        self.pub.publish(self.path_msg)
        self.timer = self.create_timer(2.0, self.timer_cb)


    # - REPUBLISH CALLBACK FUNCTION - 
    # Refresh time stamps and republishes path.
    # Fixes Rviz displaying issues.
    def timer_cb(self):
        stamp = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = stamp
        for ps in self.path_msg.poses:
            ps.header.stamp = stamp
        self.pub.publish(self.path_msg) 

    # - LOAD CONES FUNCTION - 
    # This reads the CSV file and seperates the cones.
    def load_cones(self, csv_path: Path):
        blue, yellow, orange = [], [], []
        with csv_path.open('r', newline='') as f: 
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                tag = row[0].strip()
                x = float(row[1]); y = float(row[2]) # append x and y for each tag type.
                if tag == 'blue':
                    blue.append((x, y))
                elif tag == 'yellow':
                    yellow.append((x, y))
                elif tag == 'big_orange':
                    orange.append((x, y))
        return blue, yellow, orange

    # - COMPUTE MIDPOINTS WITH CONE MUTUAL NEAREST NEIGHBOUR -
    # This pairs blue cones with their nearest yellow cones and computes the midpoint between them.
    def compute_midpoints_mutual_nn(self, blue, yellow):
        if not blue or not yellow:
            return []

        # Nearest yellow for each blue cone.
        b_to_y = {}
        for i, b in enumerate(blue):
            j_best = min(range(len(yellow)), key=lambda j: dist2(b, yellow[j]))
            b_to_y[i] = j_best

        # Nearest blue for each yellow cone.
        y_to_b = {}
        for j, y in enumerate(yellow):
            i_best = min(range(len(blue)), key=lambda i: dist2(y, blue[i]))
            y_to_b[j] = i_best

        midpoints = []

        # Keep only the matches who are mutual.
        for i, j in b_to_y.items():
            if y_to_b.get(j, None) == i:   
                bx, by = blue[i]
                yx, yy = yellow[j]
                midpoints.append(((bx + yx) * 0.5, (by + yy) * 0.5))

        return midpoints
    
    # - CLOSE LOOP HELPER FUNCTION -
    # Close loop helper. Simly ensures last point equals first.
    def close_loop(self, pts, eps=1e-6):
        if not pts:
            return pts
        first = pts[0]
        last = pts[-1]
        
        # Compare distance between first and last
        if math.hypot(first[0]-last[0], first[1]-last[1]) > eps:
            pts = list(pts) + [first]
        return pts

    # - PATH SMOOTHING -
    # This essentially slices every corner of a polygon to get a smoother curve.
    # Improves MPC accuracy by fixing jagged lines in the trajectory path.
    # Chaikin's closed loop algorithm.
    def smooth_path(self, pts, iterations=3):

        if len(pts) < 4:
            return pts

        pts = list(pts)

        # Ensure closed loop.
        pts = self.close_loop(pts)

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
        closed = (math.hypot(pts[0][0]-pts[-1][0], pts[0][1]-pts[-1][1]) < 1e-9)
        if not closed:
            pts.append(pts[0])
            closed = True

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
        if out and (math.hypot(out[0][0]-out[-1][0], out[0][1]-out[-1][1]) > 1e-6):
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

    # - MIDPOINT ORDERING FUNCTION -
    # This turns what would be an unordered list of midpoints into a continuous loop using greedy ordering.
    def order_points_nearest_neighbour(self, points):
        if len(points) < 3:
            return points

        pts = list(points)

        # Start from lowest x, then lowest y.
        start_idx = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
        ordered = [pts.pop(start_idx)]

        while pts:
            last = ordered[-1]
            # Repeatedly choose the nearest unused.
            next_idx = min(range(len(pts)), key=lambda i: dist2(last, pts[i]))
            ordered.append(pts.pop(next_idx))

        return ordered

def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
