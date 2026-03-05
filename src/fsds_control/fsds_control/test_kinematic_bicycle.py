from fsds_control.kinematic_bicycle import VehicleParams, State, step

def main():
    p = VehicleParams()

    # Start at origin, facing +x, stationary
    s = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
    dt = 0.02  # 50 Hz

    # Accelerate straight for 5 seconds
    for _ in range(int(5.0 / dt)):
        s = step(s, steer_cmd=0.0, throttle=1.0, brake=0.0, dt=dt, params=p)

    print("After 5s full throttle straight:")
    print(s)

    # Expect v ~ accel_max * 5 = 1.8*5 = 9 m/s (roughly)
    # x should be ~ 0.5*a*t^2 = 0.5*1.8*25 = 22.5 m (roughly)
    # y should remain ~0, yaw ~0

    # Brake to a stop
    for _ in range(int(3.0 / dt)):
        s = step(s, steer_cmd=0.0, throttle=0.0, brake=1.0, dt=dt, params=p)

    print("\nAfter 3s full brake:")
    print(s)

    # Expect v to clamp to 0 and not go negative

    # Turn left at constant speed (no accel/brake)
    s = State(x=0.0, y=0.0, yaw=0.0, v=10.0)
    for _ in range(int(2.0 / dt)):
        s = step(s, steer_cmd=0.5, throttle=0.0, brake=0.0, dt=dt, params=p)

    print("\nAfter 2s turning (steer_cmd=0.5) at v=10 m/s:")
    print(s)

if __name__ == "__main__":
    main()