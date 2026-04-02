#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

"""
- EVALUATION LOGGER - 

Builds a rather useful set of CSVs for evaluation.

Launch with:
ros2 run fsds_control evaluation_logger

Output file is hard coded to home\mpc_eval 
"""


class EvaluationLogger(Node):
    def __init__(self):
        super().__init__('evaluation_logger')

        self.output_dir = os.path.join(os.path.expanduser('~'), 'mpc_eval')
        os.makedirs(self.output_dir, exist_ok=True)

        self.run_id = self._next_run_id()

        self.samples_labelled_csv_path = os.path.join(
            self.output_dir, f'{self.run_id}_samples_labelled.csv'
        )
        self.laps_csv_path = os.path.join(
            self.output_dir, f'{self.run_id}_laps.csv'
        )
        self.summary_csv_path = os.path.join(
            self.output_dir, 'summary.csv'
        )

        self.samples = []
        self.estop_seen = False

        self.prev_waypoint_idx = None
        self.lap_count = 0
        self.seen_far_along_track = False
        self.lap_start_sample_idx = 0
        self.lap_boundaries = []  
        self.lap_detection_armed = False

        self.first_timestamp_sec = None

        self.create_subscription(String, '/mpc_debug', self._debug_cb, 100)

        self.get_logger().info(f'Logging to {self.output_dir}')
        self.get_logger().info(f'Run ID: {self.run_id}')

    def _next_run_id(self) -> str:
        used = set()

        for name in os.listdir(self.output_dir):
            if name.endswith('_samples_labelled.csv'):
                prefix = name[:-len('_samples_labelled.csv')]
                if prefix.isdigit():
                    used.add(int(prefix))
            elif name.endswith('_laps.csv'):
                prefix = name[:-len('_laps.csv')]
                if prefix.isdigit():
                    used.add(int(prefix))

        run_num = 1
        while run_num in used:
            run_num += 1

        return str(run_num)

    def _debug_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        ts = data.get('timestamp_sec', None)
        if ts is not None and self.first_timestamp_sec is None:
            self.first_timestamp_sec = float(ts)
            self.lap_start_time = float(ts)

        data['elapsed_time_s'] = (
            float(ts) - self.first_timestamp_sec
            if ts is not None and self.first_timestamp_sec is not None
            else None
        )

        speed = float(data.get('speed_mps', 0.0) or 0.0)
        ref_speed = float(data.get('ref_speed_mps', 0.0) or 0.0)
        cte = float(data.get('cte_m', 0.0) or 0.0)
        heading_deg = float(data.get('heading_error_deg', 0.0) or 0.0)
        steer = float(data.get('steer_cmd', 0.0) or 0.0)

        data['speed_error_mps'] = speed - ref_speed
        data['abs_cte_m'] = abs(cte)
        data['abs_heading_error_deg'] = abs(heading_deg)
        data['abs_steer_cmd'] = abs(steer)
        data['solver_failed'] = not bool(data.get('solver_success', True))
        data['lap_event'] = ''
        data['lap_number'] = self.lap_count

        if data.get('estop_active', False):
            self.estop_seen = True

        idx = data.get('waypoint_idx', None)
        n_points = data.get('path_length_points', None)

        if idx is not None and n_points is not None and int(n_points) > 10:
            idx = int(idx)
            n_points = int(n_points)

            if idx > 0.8 * n_points:
                self.seen_far_along_track = True

            crossed_start = (
                self.prev_waypoint_idx is not None
                and self.prev_waypoint_idx > 0.8 * n_points
                and idx < 0.2 * n_points
                and speed > 2.0
            )

            if crossed_start and not self.lap_detection_armed:
                self.lap_detection_armed = True
                self.seen_far_along_track = False
                self.lap_start_sample_idx = len(self.samples) + 1
                self.get_logger().info('Lap 1 started')

            elif crossed_start and self.lap_detection_armed and self.seen_far_along_track:
                completed_lap_number = self.lap_count + 1
                lap_end_sample_idx = len(self.samples)

                self.lap_boundaries.append({
                    'lap_number': completed_lap_number,
                    'start_sample_idx': self.lap_start_sample_idx,
                    'end_sample_idx': lap_end_sample_idx,
                })

                self.lap_count += 1
                data['lap_event'] = 'lap_complete'
                data['lap_number'] = self.lap_count

                self.get_logger().info(
                    f'Lap {self.lap_count} completed'
                )

                self.seen_far_along_track = False
                self.lap_start_sample_idx = len(self.samples) + 1

            self.prev_waypoint_idx = idx

        self.samples.append(data)

    def write_results(self):
        if not self.samples:
            self.get_logger().warn('No data collected.')
            return

        self._write_labelled_samples_csv()
        lap_rows = self._compute_lap_rows()
        self._write_laps_csv(lap_rows)
        self._append_summary_csv(lap_rows)

        self.get_logger().info(f'Wrote labelled samples: {self.samples_labelled_csv_path}')
        self.get_logger().info(f'Wrote lap analysis: {self.laps_csv_path}')
        self.get_logger().info(f'Updated run summary: {self.summary_csv_path}')

    def _write_labelled_samples_csv(self):
        fieldnames = [
            'run_id',
            'timestamp_sec',
            'elapsed_time_s',
            'lap_number',
            'lap_event',
            'waypoint_idx',
            'path_length_points',
            'x',
            'y',
            'yaw_rad',
            'yaw_deg',
            'speed_mps',
            'ref_speed_mps',
            'speed_error_mps',
            'cte_m',
            'abs_cte_m',
            'heading_error_rad',
            'heading_error_deg',
            'abs_heading_error_deg',
            'steer_cmd',
            'abs_steer_cmd',
            'throttle_cmd',
            'brake_cmd',
            'solve_time_ms',
            'tick_time_ms',
            'solver_success',
            'solver_failed',
            'solver_status',
            'solver_iters',
            'deadline_miss',
            'estop_active',
        ]

        with open(self.samples_labelled_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.samples:
                row = {'run_id': self.run_id}
                for key in fieldnames:
                    if key == 'run_id':
                        continue
                    row[key] = s.get(key, None)
                writer.writerow(row)

    def _compute_lap_rows(self):
        lap_rows = []
        for lap_meta in self.lap_boundaries:
            start_idx = lap_meta['start_sample_idx']
            end_idx = lap_meta['end_sample_idx']
            lap_samples = self.samples[start_idx:end_idx]

            if not lap_samples:
                continue

            cte = [float(s.get('cte_m', 0.0) or 0.0) for s in lap_samples]
            heading = [float(s.get('heading_error_deg', 0.0) or 0.0) for s in lap_samples]
            speed = [float(s.get('speed_mps', 0.0) or 0.0) for s in lap_samples]
            ref_speed = [float(s.get('ref_speed_mps', 0.0) or 0.0) for s in lap_samples]
            steer = [float(s.get('steer_cmd', 0.0) or 0.0) for s in lap_samples]
            solve = [
                float(s['solve_time_ms']) for s in lap_samples
                if s.get('solve_time_ms') is not None
            ]
            tick = [
                float(s['tick_time_ms']) for s in lap_samples
                if s.get('tick_time_ms') is not None
            ]

            speed_error = [v - rv for v, rv in zip(speed, ref_speed)]
            deadline_misses = sum(1 for s in lap_samples if bool(s.get('deadline_miss', False)))
            solver_failures = sum(1 for s in lap_samples if not bool(s.get('solver_success', True)))
            estop_during_lap = any(bool(s.get('estop_active', False)) for s in lap_samples)

            lap_rows.append({
                'run_id': self.run_id,
                'lap_number': lap_meta['lap_number'],
                'lap_sample_count': len(lap_samples),
                'rms_cross_track_error_m': self._rms(cte),
                'mean_abs_cross_track_error_m': self._mean_abs(cte),
                'max_abs_cross_track_error_m': self._max_abs(cte),
                'rms_heading_error_deg': self._rms(heading),
                'mean_abs_heading_error_deg': self._mean_abs(heading),
                'max_abs_heading_error_deg': self._max_abs(heading),
                'rms_speed_error_mps': self._rms(speed_error),
                'mean_abs_speed_error_mps': self._mean_abs(speed_error),
                'max_abs_speed_error_mps': self._max_abs(speed_error),
                'mean_speed_mps': self._mean(speed),
                'max_speed_mps': max(speed) if speed else 0.0,
                'mean_abs_steering_command': self._mean_abs(steer),
                'max_abs_steering_command': self._max_abs(steer),
                'mean_solver_time_ms': self._mean(solve),
                'max_solver_time_ms': max(solve) if solve else 0.0,
                'mean_tick_time_ms': self._mean(tick),
                'max_tick_time_ms': max(tick) if tick else 0.0,
                'deadline_miss_count': deadline_misses,
                'solver_failure_count': solver_failures,
                'estop_during_lap': estop_during_lap,
            })

        return lap_rows

    def _write_laps_csv(self, lap_rows):
        fieldnames = [
            'run_id',
            'lap_number',
            'lap_sample_count',
            'rms_cross_track_error_m',
            'mean_abs_cross_track_error_m',
            'max_abs_cross_track_error_m',
            'rms_heading_error_deg',
            'mean_abs_heading_error_deg',
            'max_abs_heading_error_deg',
            'rms_speed_error_mps',
            'mean_abs_speed_error_mps',
            'max_abs_speed_error_mps',
            'mean_speed_mps',
            'max_speed_mps',
            'mean_abs_steering_command',
            'max_abs_steering_command',
            'mean_solver_time_ms',
            'max_solver_time_ms',
            'mean_tick_time_ms',
            'max_tick_time_ms',
            'deadline_miss_count',
            'solver_failure_count',
            'estop_during_lap',
        ]

        with open(self.laps_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in lap_rows:
                writer.writerow(row)

    def _append_summary_csv(self, lap_rows):
        cte = [float(s.get('cte_m', 0.0) or 0.0) for s in self.samples]
        heading = [float(s.get('heading_error_deg', 0.0) or 0.0) for s in self.samples]
        speed = [float(s.get('speed_mps', 0.0) or 0.0) for s in self.samples]
        ref_speed = [float(s.get('ref_speed_mps', 0.0) or 0.0) for s in self.samples]
        steer = [float(s.get('steer_cmd', 0.0) or 0.0) for s in self.samples]
        solve = [
            float(s['solve_time_ms']) for s in self.samples
            if s.get('solve_time_ms') is not None
        ]
        tick = [
            float(s['tick_time_ms']) for s in self.samples
            if s.get('tick_time_ms') is not None
        ]
        speed_error = [v - rv for v, rv in zip(speed, ref_speed)]

        timestamps = [
            float(s['timestamp_sec']) for s in self.samples
            if s.get('timestamp_sec') is not None
        ]
        run_duration_s = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0

        deadline_misses = sum(1 for s in self.samples if bool(s.get('deadline_miss', False)))
        solver_failures = sum(1 for s in self.samples if not bool(s.get('solver_success', True)))

        summary_row = {
            'run_id': self.run_id,
            'total_sample_count': len(self.samples),
            'completed_lap_count': len(lap_rows),
            'run_duration_s': run_duration_s,
            'rms_cross_track_error_m': self._rms(cte),
            'mean_abs_cross_track_error_m': self._mean_abs(cte),
            'max_abs_cross_track_error_m': self._max_abs(cte),
            'rms_heading_error_deg': self._rms(heading),
            'mean_abs_heading_error_deg': self._mean_abs(heading),
            'max_abs_heading_error_deg': self._max_abs(heading),
            'rms_speed_error_mps': self._rms(speed_error),
            'mean_abs_speed_error_mps': self._mean_abs(speed_error),
            'max_abs_speed_error_mps': self._max_abs(speed_error),
            'mean_speed_mps': self._mean(speed),
            'max_speed_mps': max(speed) if speed else 0.0,
            'mean_abs_steering_command': self._mean_abs(steer),
            'max_abs_steering_command': self._max_abs(steer),
            'mean_solver_time_ms': self._mean(solve),
            'max_solver_time_ms': max(solve) if solve else 0.0,
            'mean_tick_time_ms': self._mean(tick),
            'max_tick_time_ms': max(tick) if tick else 0.0,
            'deadline_miss_count': deadline_misses,
            'solver_failure_count': solver_failures,
            'estop_triggered': self.estop_seen,
        }

        fieldnames = list(summary_row.keys())
        file_exists = os.path.exists(self.summary_csv_path)

        with open(self.summary_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary_row)

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _mean_abs(values):
        return sum(abs(v) for v in values) / len(values) if values else 0.0

    @staticmethod
    def _max_abs(values):
        return max((abs(v) for v in values), default=0.0)

    @staticmethod
    def _rms(values):
        return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def main():
    rclpy.init()
    node = EvaluationLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_results()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()