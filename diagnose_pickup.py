#!/usr/bin/env python3
"""Diagnostic script to test pickup mechanics"""

import numpy as np
import yaml
from warehouse_env import WarehouseEnv

def test_pickup():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = WarehouseEnv(config)
    obs = env.reset()

    print("=" * 60)
    print("PICKUP DIAGNOSTIC TEST")
    print("=" * 60)

    # Print initial state
    print(f"\nGrid size: {env.grid_size}")
    print(f"Num robots: {env.num_robots}")
    print(f"Num packages: {len(env.packages)}")
    print(f"Max speed: {env.max_speed}")
    print(f"sparse_rewards: {env.sparse_rewards}")

    # Print robot positions
    print("\n--- Robot Initial Positions ---")
    for robot in env.robots:
        print(f"  Robot {robot.id}: pos={robot.position}, speed={robot.speed:.2f}")

    # Print package positions (first 5)
    print("\n--- Package Positions (first 5) ---")
    for pkg in env.packages[:5]:
        print(f"  Package {pkg.id}: pos={pkg.position}, dest={pkg.destination}, assigned={pkg.assigned_robot}")

    # Print station positions
    print("\n--- Station Positions ---")
    for station in env.stations:
        print(f"  {station.type}: {station.position}")

    # Calculate distance from robot 0 to nearest package
    robot = env.robots[0]
    min_dist = float('inf')
    for pkg in env.packages:
        if not pkg.is_delivered and pkg.assigned_robot is None:
            dist = np.linalg.norm(robot.position - pkg.position)
            if dist < min_dist:
                min_dist = dist
    print(f"\n--- Distance robot 0 to nearest package: {min_dist:.2f} ---")

    # Now manually test pickup by moving robot 0 to a package
    print("\n" + "=" * 60)
    print("TEST 1: Manual pickup test - move robot to package")
    print("=" * 60)

    # Get first available package
    target_pkg = None
    for pkg in env.packages:
        if not pkg.is_delivered and pkg.assigned_robot is None:
            target_pkg = pkg
            break

    if target_pkg:
        # Force robot to be at package position
        robot.position = target_pkg.position.copy()
        robot.velocity = np.zeros(2)
        print(f"Moved robot 0 to package position: {robot.position}")
        print(f"Robot speed: {robot.speed}")
        print(f"Target package: {target_pkg.id} at {target_pkg.position}")

        # Now call step with zero action (should trigger auto-pickup)
        action = np.array([0.0, 0.0, 0.0])  # Stop and no gripper
        actions_dict = {i: action for i in range(env.num_robots)}

        print(f"\nCalling env.step() with action: {action}")
        obs, rewards, dones, info = env.step(actions_dict)

        print(f"Robot 0 carrying_package: {robot.carrying_package}")
        print(f"Robot 0 reward: {rewards[0]}")
        print(f"Packages delivered: {env.stats['packages_delivered']}")

        if robot.carrying_package is not None:
            print("SUCCESS: Robot picked up package!")
        else:
            print("FAILURE: Robot did NOT pick up package!")
            # Check why
            print(f"  Robot speed: {robot.speed}")
            print(f"  Distance to package: {np.linalg.norm(robot.position - target_pkg.position)}")
            print(f"  Package assigned_robot: {target_pkg.assigned_robot}")
            print(f"  Package is_delivered: {target_pkg.is_delivered}")

    # Test 2: Test with gripper action
    print("\n" + "=" * 60)
    print("TEST 2: Test with gripper action = 1.0")
    print("=" * 60)

    env.reset()
    robot = env.robots[0]

    target_pkg = None
    for pkg in env.packages:
        if not pkg.is_delivered and pkg.assigned_robot is None:
            target_pkg = pkg
            break

    if target_pkg:
        robot.position = target_pkg.position.copy()
        robot.velocity = np.zeros(2)

        action = np.array([0.0, 0.0, 1.0])  # Stop with gripper active
        actions_dict = {i: action for i in range(env.num_robots)}

        print(f"Calling env.step() with gripper=1.0")
        obs, rewards, dones, info = env.step(actions_dict)

        print(f"Robot 0 carrying_package: {robot.carrying_package}")
        if robot.carrying_package is not None:
            print("SUCCESS: Robot picked up package with gripper!")
        else:
            print("FAILURE: Robot did NOT pick up package with gripper!")

    # Test 3: Full delivery cycle
    print("\n" + "=" * 60)
    print("TEST 3: Full delivery cycle")
    print("=" * 60)

    env.reset()
    robot = env.robots[0]

    # Step 1: Move to package
    target_pkg = env.packages[0]
    robot.position = target_pkg.position.copy()
    robot.velocity = np.zeros(2)

    # Pick up
    action = np.array([0.0, 0.0, 1.0])
    actions_dict = {i: action for i in range(env.num_robots)}
    obs, rewards, dones, info = env.step(actions_dict)

    print(f"After pickup step:")
    print(f"  carrying_package: {robot.carrying_package}")
    print(f"  reward: {rewards[0]}")

    if robot.carrying_package is not None:
        # Step 2: Move to destination
        pkg = env._get_package(robot.carrying_package)
        print(f"  Package destination: {pkg.destination}")

        # Move robot to destination
        robot.position = pkg.destination.copy()
        robot.velocity = np.zeros(2)

        # Step to trigger delivery
        obs, rewards, dones, info = env.step(actions_dict)

        print(f"After delivery step:")
        print(f"  carrying_package: {robot.carrying_package}")
        print(f"  reward: {rewards[0]}")
        print(f"  packages_delivered: {env.stats['packages_delivered']}")

        if robot.carrying_package is None and env.stats['packages_delivered'] > 0:
            print("SUCCESS: Full delivery cycle works!")
        else:
            print("FAILURE: Delivery did not complete!")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_pickup()
