"""Diagnostic script to verify environment mechanics work correctly"""

import numpy as np
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

from warehouse_env import WarehouseEnv

def main():
    env = WarehouseEnv(config)
    obs = env.reset()

    print('=' * 60)
    print('ENVIRONMENT DIAGNOSTIC')
    print('=' * 60)
    print(f'Grid size: {env.grid_size}')
    print(f'Robots: {len(env.robots)}')
    print(f'Packages: {len(env.packages)}')
    print()

    # Check pickup/delivery stations
    pickup = [s for s in env.stations if s.type == "pickup"]
    delivery = [s for s in env.stations if "delivery" in s.type]
    print(f'Pickup station: {pickup[0].position if pickup else "NONE"}')
    print(f'Delivery stations: {[s.position for s in delivery]}')
    print()

    # Check where packages spawn
    print('Package positions (first 5):')
    for pkg in env.packages[:5]:
        print(f'  pkg {pkg.id}: pos=[{pkg.position[0]:.1f}, {pkg.position[1]:.1f}], '
              f'dest=[{pkg.destination[0]:.1f}, {pkg.destination[1]:.1f}]')
    print()

    # Check where robots spawn
    print('Robot starting positions:')
    for robot in env.robots[:4]:
        print(f'  robot {robot.id}: pos=[{robot.position[0]:.1f}, {robot.position[1]:.1f}]')
    print()

    # Test: can a robot navigate to a package?
    print('-' * 60)
    print('TEST: Manual navigation to package')
    print('-' * 60)

    robot = env.robots[0]
    pkg = env.packages[0]
    initial_dist = np.linalg.norm(pkg.position - robot.position)
    print(f'Robot 0 at [{robot.position[0]:.1f}, {robot.position[1]:.1f}]')
    print(f'Target pkg at [{pkg.position[0]:.1f}, {pkg.position[1]:.1f}]')
    print(f'Initial distance: {initial_dist:.1f}')
    print()

    reached = False
    for step in range(150):
        direction = pkg.position - robot.position
        dist = np.linalg.norm(direction)

        if dist < 1.5:
            print(f'Step {step}: REACHED package! dist={dist:.2f}')
            reached = True
            break

        # Simple navigation controller
        angle_to_pkg = np.arctan2(direction[1], direction[0])
        angle_diff = angle_to_pkg - robot.orientation
        while angle_diff > np.pi: angle_diff -= 2 * np.pi
        while angle_diff < -np.pi: angle_diff += 2 * np.pi

        if abs(angle_diff) > 0.3:
            action = np.array([0.0, np.clip(angle_diff * 2, -1, 1), 0])  # rotate
        else:
            action = np.array([1.0, angle_diff * 0.5, 0])  # move forward

        actions = {i: np.zeros(3, dtype=np.float32) for i in range(8)}
        actions[0] = action.astype(np.float32)

        next_obs, rewards, dones, info = env.step(actions)
        robot = env.robots[0]

        if step % 30 == 0:
            print(f'  Step {step}: pos=[{robot.position[0]:.1f}, {robot.position[1]:.1f}], '
                  f'dist={dist:.1f}, speed={robot.speed:.2f}, reward={rewards[0]:.2f}')

    print()
    print(f'Reached package: {reached}')
    print(f'Robot carrying: {robot.carrying_package}')
    print(f'Robot speed: {robot.speed:.2f} (need < 1.0 to pickup)')
    print()

    if not reached:
        print('ERROR: Robot could not reach package with simple controller!')
        return

    # Wait for speed to drop and try pickup
    print('Waiting for slowdown...')
    for step in range(20):
        actions = {i: np.zeros(3, dtype=np.float32) for i in range(8)}
        actions[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # stop
        env.step(actions)
        robot = env.robots[0]
        if robot.speed < 0.5:
            break

    print(f'Speed after stopping: {robot.speed:.2f}')
    print(f'Carrying package: {robot.carrying_package}')
    print()

    # Check package stats
    print('Environment stats:', env.stats)
    print()

    if robot.carrying_package is not None:
        print('SUCCESS: Package pickup works!')

        # Now try to deliver
        print()
        print('-' * 60)
        print('TEST: Navigate to delivery')
        print('-' * 60)
        pkg = env._get_package(robot.carrying_package)
        print(f'Package destination: {pkg.destination}')

        for step in range(200):
            direction = pkg.destination - robot.position
            dist = np.linalg.norm(direction)

            if dist < 1.5:
                print(f'Step {step}: REACHED delivery! dist={dist:.2f}')
                break

            angle_to_dest = np.arctan2(direction[1], direction[0])
            angle_diff = angle_to_dest - robot.orientation
            while angle_diff > np.pi: angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi

            if abs(angle_diff) > 0.3:
                action = np.array([0.0, np.clip(angle_diff * 2, -1, 1), 0])
            else:
                action = np.array([1.0, angle_diff * 0.5, 0])

            actions = {i: np.zeros(3, dtype=np.float32) for i in range(8)}
            actions[0] = action.astype(np.float32)
            env.step(actions)
            robot = env.robots[0]

            if step % 40 == 0:
                print(f'  Step {step}: dist={dist:.1f}')

        print(f'Final carrying: {robot.carrying_package}')
        print(f'Packages delivered: {env.stats["packages_delivered"]}')

        if env.stats['packages_delivered'] > 0:
            print()
            print('SUCCESS: Full delivery cycle works!')
        else:
            print()
            print('ISSUE: Reached destination but no delivery recorded')
    else:
        print('ISSUE: Pickup did not work even when robot was close and stopped')


if __name__ == '__main__':
    main()
