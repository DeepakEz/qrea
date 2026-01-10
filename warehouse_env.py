"""
Warehouse Environment for QREA Multi-Robot System
Complete 3D physics simulation with packages, robots, and logistics
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import copy


class PackagePriority(Enum):
    STANDARD = 1
    EXPRESS = 2
    FRAGILE = 3


@dataclass
class Package:
    """Package representation"""
    id: int
    position: np.ndarray
    destination: np.ndarray
    weight: float
    priority: PackagePriority
    reward: float
    spawn_time: float
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    assigned_robot: Optional[int] = None
    
    @property
    def waiting_time(self) -> float:
        if self.delivery_time is not None:
            return self.delivery_time - self.spawn_time
        return 0.0
    
    @property
    def is_delivered(self) -> bool:
        return self.delivery_time is not None


@dataclass
class Robot:
    """Robot state representation"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    orientation: float  # radians
    angular_velocity: float
    battery: float
    carrying_package: Optional[int] = None
    target_package: Optional[int] = None
    target_station: Optional[int] = None
    collision_count: int = 0
    distance_traveled: float = 0.0
    packages_delivered: int = 0
    energy_consumed: float = 0.0
    
    @property
    def is_carrying(self) -> bool:
        return self.carrying_package is not None
    
    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


@dataclass
class Station:
    """Charging/pickup/delivery station"""
    id: int
    position: np.ndarray
    type: str  # "pickup", "delivery", "charging"
    capacity: int
    occupancy: int = 0


class WarehouseEnv(gym.Env):
    """
    Multi-robot warehouse environment
    
    Observation Space:
    - Robot state: position, velocity, orientation, battery, carrying
    - LIDAR: distance measurements
    - Package info: positions, priorities, distances
    - Station info: positions, occupancy
    - Other robots: relative positions, velocities
    - Field state: UPRT field values at robot position
    
    Action Space:
    - Linear velocity: [-max_speed, max_speed]
    - Angular velocity: [-max_omega, max_omega]
    - Gripper: [0, 1] (release, grab)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        env_cfg = config['environment']
        robot_cfg = env_cfg['robot']
        sensor_cfg = env_cfg['sensors']
        
        # Environment parameters
        self.grid_size = np.array(env_cfg['grid_size'], dtype=np.float32)
        self.num_robots = env_cfg['num_robots']
        self.num_packages = env_cfg['num_packages']
        self.dt = env_cfg['dt']
        self.max_steps = env_cfg['max_episode_steps']
        
        # Robot parameters
        self.robot_radius = robot_cfg['radius']
        self.max_speed = robot_cfg['max_speed']
        self.max_acceleration = robot_cfg['max_acceleration']
        self.max_angular_vel = robot_cfg['max_angular_velocity']
        self.battery_capacity = robot_cfg['battery_capacity']
        self.battery_drain = {
            'idle': robot_cfg['battery_drain_idle'],
            'moving': robot_cfg['battery_drain_moving'],
            'carrying': robot_cfg['battery_drain_carrying']
        }
        self.charging_rate = robot_cfg['charging_rate']
        
        # Sensor parameters
        self.lidar_rays = sensor_cfg['lidar']['num_rays']
        self.lidar_range = sensor_cfg['lidar']['max_range']
        self.lidar_noise = sensor_cfg['lidar']['noise_std']
        
        # Package parameters
        self.package_spawn_rate = env_cfg['package_spawn_rate']
        self.package_types = env_cfg['package_types']
        
        # Initialize stations
        self.stations = self._init_stations(env_cfg['stations'])
        
        # State
        self.robots: List[Robot] = []
        self.packages: List[Package] = []
        self.obstacles: List[np.ndarray] = []
        self.current_step = 0
        self.episode_reward = 0.0
        self.next_package_id = 0

        # Task variant configuration (set by TaskVariant.apply_*)
        self.package_spawn_zones: Optional[List[Dict]] = None  # Clustered spawning
        self.dynamic_priorities: bool = False  # Dynamic priority changes

        # Reward configuration
        # sparse_rewards=True: Only reward delivery/pickup, high collision penalty (harder, more meaningful)
        # sparse_rewards=False: Dense progress shaping (easier, for initial learning)
        self.sparse_rewards: bool = env_cfg.get('sparse_rewards', False)

        # Spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
        # Statistics
        self.stats = {
            'packages_delivered': 0,
            'packages_picked_up': 0,  # Track pickups to show learning progress
            'total_waiting_time': 0.0,
            'total_distance': 0.0,
            'total_energy': 0.0,
            'collisions': 0,
            'timeouts': 0
        }
        
    def _init_stations(self, station_cfg: dict) -> List[Station]:
        """Initialize pickup/delivery/charging stations"""
        stations = []
        for i, (stype, pos) in enumerate(station_cfg.items()):
            stations.append(Station(
                id=i,
                position=np.array(pos, dtype=np.float32),
                type=stype,
                capacity=self.num_robots
            ))
        return stations
    
    def _create_action_space(self):
        """Create continuous action space for each robot"""
        # Actions: [linear_vel, angular_vel, gripper]
        return spaces.Box(
            low=np.array([-self.max_speed, -self.max_angular_vel, 0.0]),
            high=np.array([self.max_speed, self.max_angular_vel, 1.0]),
            dtype=np.float32
        )
    
    def _create_observation_space(self):
        """Create observation space"""
        # Robot state (10) + LIDAR (lidar_rays) + Package info (100*4) + Station info (4*3) + Others ((num_robots-1)*8)
        others_dim = (self.num_robots - 1) * 8  # Adapts to actual number of robots
        obs_dim = 10 + self.lidar_rays + 400 + 12 + others_dim
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.next_package_id = 0
        self._just_delivered = {}  # Track deliveries for reward calculation
        
        # Reset robots
        self.robots = []
        # Spawn robots in working zone (central area, reasonable distance to all stations)
        center = self.grid_size / 2
        spawn_radius = min(self.grid_size) * 0.3  # 30% of grid size
        for i in range(self.num_robots):
            # Spawn in circular zone around center
            angle = 2 * np.pi * i / self.num_robots
            r = spawn_radius * (0.5 + 0.5 * np.random.random())
            pos = np.array([
                center[0] + r * np.cos(angle),
                center[1] + r * np.sin(angle)
            ], dtype=np.float32)
            pos = np.clip(pos, 2, self.grid_size - 2)
            
            self.robots.append(Robot(
                id=i,
                position=pos,
                velocity=np.zeros(2, dtype=np.float32),
                orientation=np.random.uniform(0, 2*np.pi),
                angular_velocity=0.0,
                battery=self.battery_capacity
            ))
        
        # Reset packages
        self.packages = []
        self._spawn_packages(initial=True)
        
        # Reset statistics
        self.stats = {k: 0.0 if isinstance(v, float) else 0 for k, v in self.stats.items()}
        
        # Get initial observations
        observations = {}
        for robot in self.robots:
            observations[robot.id] = self._get_observation(robot)
        
        return observations
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],        # rewards
        Dict[int, bool],         # dones
        Dict[int, dict]          # info
    ]:
        """Execute one environment step"""
        self.current_step += 1

        # Apply actions to robots (randomize order to prevent robot 0 priority)
        robot_order = np.random.permutation(len(self.robots))
        for idx in robot_order:
            robot = self.robots[idx]
            if robot.id in actions:
                self._apply_action(robot, actions[robot.id])
        
        # Update physics
        self._update_physics()
        
        # Update battery
        self._update_battery()
        
        # Handle package pickup/delivery
        self._handle_packages()
        
        # Spawn new packages
        if np.random.random() < self.package_spawn_rate * self.dt:
            self._spawn_packages()
        
        # Check collisions
        self._check_collisions()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Get observations
        observations = {}
        for robot in self.robots:
            observations[robot.id] = self._get_observation(robot)
        
        # Check termination
        dones = {}
        info = {}
        terminated = self.current_step >= self.max_steps
        
        for robot in self.robots:
            dones[robot.id] = terminated or robot.battery <= 0
            info[robot.id] = self._get_info(robot)
        
        dones['__all__'] = terminated
        
        return observations, rewards, dones, info
    
    def _apply_action(self, robot: Robot, action: np.ndarray):
        """Apply action to robot"""
        linear_vel, angular_vel, gripper = action
        
        # Clip actions
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        # Update velocity (with acceleration limits)
        target_vel = np.array([
            linear_vel * np.cos(robot.orientation),
            linear_vel * np.sin(robot.orientation)
        ])
        
        vel_diff = target_vel - robot.velocity
        max_vel_change = self.max_acceleration * self.dt
        
        if np.linalg.norm(vel_diff) > max_vel_change:
            vel_diff = vel_diff / np.linalg.norm(vel_diff) * max_vel_change
        
        robot.velocity = robot.velocity + vel_diff
        
        # Update angular velocity
        robot.angular_velocity = angular_vel
        
        # Handle gripper - agent must LEARN to use gripper action
        # No auto-pickup: research validity requires natural learning
        should_try_pickup = gripper > 0.5

        if should_try_pickup:
            self._try_pickup_or_deliver(robot)
    
    def _update_physics(self):
        """Update robot positions and orientations"""
        for robot in self.robots:
            # Update position
            new_pos = robot.position + robot.velocity * self.dt
            
            # Boundary checking
            new_pos = np.clip(new_pos, 
                            [self.robot_radius, self.robot_radius],
                            self.grid_size - self.robot_radius)
            
            # Track distance
            robot.distance_traveled += np.linalg.norm(new_pos - robot.position)
            robot.position = new_pos
            
            # Update orientation
            robot.orientation += robot.angular_velocity * self.dt
            robot.orientation = robot.orientation % (2 * np.pi)
            
            # Update package position if carrying
            if robot.carrying_package is not None:
                pkg = self._get_package(robot.carrying_package)
                if pkg:
                    pkg.position = robot.position.copy()
    
    def _update_battery(self):
        """Update robot battery levels"""
        for robot in self.robots:
            if robot.speed < 0.1:
                drain = self.battery_drain['idle']
            elif robot.is_carrying:
                drain = self.battery_drain['carrying']
            else:
                drain = self.battery_drain['moving']
            
            robot.battery -= drain * self.dt
            robot.energy_consumed += drain * self.dt
            
            # Check if at charging station
            for station in self.stations:
                if station.type.startswith("charging"):
                    dist = np.linalg.norm(robot.position - station.position)
                    if dist < 2.0:  # Within charging range
                        robot.battery = min(
                            robot.battery + self.charging_rate * self.dt,
                            self.battery_capacity
                        )

    def _handle_packages(self):
        """Handle package pickup and delivery"""
        # Track just-delivered packages for reward calculation
        self._just_delivered = {}  # robot_id -> package_reward

        # Randomize order to prevent robot 0 priority for deliveries
        robot_order = np.random.permutation(len(self.robots))
        for idx in robot_order:
            robot = self.robots[idx]
            if robot.carrying_package is not None:
                pkg = self._get_package(robot.carrying_package)
                if pkg and not pkg.is_delivered:
                    # Check if at destination
                    dist = np.linalg.norm(robot.position - pkg.destination)
                    if dist < 1.5:  # Within delivery range
                        pkg.delivery_time = self.current_step * self.dt
                        self._just_delivered[robot.id] = pkg.reward  # Track for reward
                        robot.carrying_package = None
                        robot.packages_delivered += 1
                        self.stats['packages_delivered'] += 1
                        self.stats['total_waiting_time'] += pkg.waiting_time
    
    def _try_pickup_or_deliver(self, robot: Robot):
        """Try to pick up or deliver a package"""
        if robot.carrying_package is not None:
            return  # Already carrying

        # Robot must slow down to pickup - ORIGINAL threshold for research validity
        if robot.speed > 1.0:
            return

        # Find nearest unassigned package - ORIGINAL radius for research validity
        min_dist = float('inf')
        nearest_pkg = None

        for pkg in self.packages:
            if pkg.is_delivered or pkg.assigned_robot is not None:
                continue

            dist = np.linalg.norm(robot.position - pkg.position)
            if dist < 1.5 and dist < min_dist:  # Original pickup radius
                min_dist = dist
                nearest_pkg = pkg
        
        if nearest_pkg:
            robot.carrying_package = nearest_pkg.id
            nearest_pkg.assigned_robot = robot.id
            nearest_pkg.pickup_time = self.current_step * self.dt
            self.stats['packages_picked_up'] += 1
            self.stats['total_distance'] += min_dist
    
    def _spawn_packages(self, initial: bool = False):
        """Spawn new packages.

        Supports task variants:
        - package_spawn_zones: If set, spawns packages in clusters
        - dynamic_priorities: If True, priorities change over episode time
        """
        num_spawn = self.num_packages if initial else 1

        for _ in range(num_spawn):
            # Random package type
            pkg_type = np.random.choice(list(self.package_types.keys()))
            pkg_config = self.package_types[pkg_type]

            # Determine spawn position
            if self.package_spawn_zones is not None:
                # Clustered spawning: pick a random zone and spawn within it
                zone = self.package_spawn_zones[np.random.randint(len(self.package_spawn_zones))]
                center = np.array(zone['center'], dtype=np.float32)
                radius = zone.get('radius', 5.0)
                spawn_pos = center + np.random.randn(2) * radius * 0.3
            else:
                # CURRICULUM FIX: Scatter packages across grid for easier discovery
                # Instead of spawning only at pickup station (too far from robots),
                # spawn packages randomly throughout the working area
                # This lets robots naturally encounter packages during exploration
                margin = 5.0  # Keep away from edges
                spawn_pos = np.array([
                    np.random.uniform(margin, self.grid_size[0] - margin),
                    np.random.uniform(margin, self.grid_size[1] - margin)
                ], dtype=np.float32)

            # Random delivery station
            delivery_stations = [s for s in self.stations if s.type.startswith("delivery")]
            dest_station = np.random.choice(delivery_stations)

            # Determine priority
            priority_val = pkg_config['priority']
            if self.dynamic_priorities:
                # Priority changes based on episode time: early = normal, late = urgent
                time_factor = self.current_step / max(1, self.max_steps)
                if time_factor > 0.7:
                    priority_val = max(priority_val, 2)  # Urgent
                elif time_factor > 0.4:
                    priority_val = max(priority_val, 1)  # High

            package = Package(
                id=self.next_package_id,
                position=spawn_pos,
                destination=dest_station.position,
                weight=pkg_config['weight'],
                priority=PackagePriority(priority_val),
                reward=pkg_config['reward'],
                spawn_time=self.current_step * self.dt
            )

            self.packages.append(package)
            self.next_package_id += 1
    
    def _check_collisions(self):
        """Check and handle robot-robot collisions"""
        for i, robot1 in enumerate(self.robots):
            for robot2 in self.robots[i+1:]:
                dist = np.linalg.norm(robot1.position - robot2.position)
                min_dist = 2 * self.robot_radius
                
                if dist < min_dist:
                    # Collision detected
                    robot1.collision_count += 1
                    robot2.collision_count += 1
                    self.stats['collisions'] += 1
                    
                    # Push apart
                    overlap = min_dist - dist
                    direction = (robot2.position - robot1.position) / (dist + 1e-6)
                    robot1.position -= direction * overlap * 0.5
                    robot2.position += direction * overlap * 0.5
    
    def _get_observation(self, robot: Robot) -> np.ndarray:
        """Get observation for a robot"""
        obs = []
        
        # Robot state (10)
        obs.extend([
            robot.position[0] / self.grid_size[0],
            robot.position[1] / self.grid_size[1],
            robot.velocity[0] / self.max_speed,
            robot.velocity[1] / self.max_speed,
            np.cos(robot.orientation),
            np.sin(robot.orientation),
            robot.angular_velocity / self.max_angular_vel,
            robot.battery / self.battery_capacity,
            1.0 if robot.is_carrying else 0.0,
            robot.speed / self.max_speed
        ])
        
        # LIDAR (32)
        lidar = self._get_lidar(robot)
        obs.extend(lidar)
        
        # Package info (100 packages * 4 features = 400)
        packages_obs = self._get_package_observations(robot, max_packages=100)
        obs.extend(packages_obs)
        
        # Station info (4 stations * 3 features = 12)
        stations_obs = self._get_station_observations(robot)
        obs.extend(stations_obs)
        
        # Other robots (7 robots * 8 features = 56)
        others_obs = self._get_other_robots_observations(robot)
        obs.extend(others_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_lidar(self, robot: Robot) -> List[float]:
        """Simulate LIDAR sensor"""
        ranges = []
        angles = np.linspace(0, 2*np.pi, self.lidar_rays, endpoint=False)
        
        for angle in angles:
            ray_angle = robot.orientation + angle
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            min_dist = self.lidar_range
            
            # Check walls
            for axis in range(2):
                for boundary in [0, self.grid_size[axis]]:
                    if abs(ray_dir[axis]) > 1e-6:
                        t = (boundary - robot.position[axis]) / ray_dir[axis]
                        if t > 0:
                            dist = t
                            if dist < min_dist:
                                min_dist = dist
            
            # Check other robots
            for other in self.robots:
                if other.id == robot.id:
                    continue
                
                to_other = other.position - robot.position
                proj = np.dot(to_other, ray_dir)
                
                if proj > 0:
                    closest_point = robot.position + ray_dir * proj
                    dist_to_ray = np.linalg.norm(other.position - closest_point)
                    
                    if dist_to_ray < self.robot_radius:
                        dist = proj - np.sqrt(self.robot_radius**2 - dist_to_ray**2)
                        if dist < min_dist and dist > 0:
                            min_dist = dist
            
            # Add noise
            min_dist += np.random.randn() * self.lidar_noise
            min_dist = np.clip(min_dist, 0, self.lidar_range)
            
            ranges.append(min_dist / self.lidar_range)
        
        return ranges
    
    def _get_package_observations(self, robot: Robot, max_packages: int) -> List[float]:
        """Get observations of nearby packages"""
        obs = []
        
        # Sort packages by distance
        packages = sorted(
            [p for p in self.packages if not p.is_delivered],
            key=lambda p: np.linalg.norm(p.position - robot.position)
        )[:max_packages]
        
        for pkg in packages:
            rel_pos = pkg.position - robot.position
            dist = np.linalg.norm(rel_pos)
            
            obs.extend([
                rel_pos[0] / self.grid_size[0],
                rel_pos[1] / self.grid_size[1],
                dist / np.linalg.norm(self.grid_size),
                pkg.priority.value / 3.0
            ])
        
        # Pad if fewer packages
        while len(obs) < max_packages * 4:
            obs.extend([0.0, 0.0, 1.0, 0.0])
        
        return obs
    
    def _get_station_observations(self, robot: Robot) -> List[float]:
        """Get observations of stations"""
        obs = []
        
        for station in self.stations:
            rel_pos = station.position - robot.position
            dist = np.linalg.norm(rel_pos)
            
            obs.extend([
                rel_pos[0] / self.grid_size[0],
                rel_pos[1] / self.grid_size[1],
                dist / np.linalg.norm(self.grid_size)
            ])
        
        return obs
    
    def _get_other_robots_observations(self, robot: Robot) -> List[float]:
        """Get observations of other robots"""
        obs = []
        
        others = [r for r in self.robots if r.id != robot.id]
        
        for other in others:
            rel_pos = other.position - robot.position
            rel_vel = other.velocity - robot.velocity
            dist = np.linalg.norm(rel_pos)
            
            obs.extend([
                rel_pos[0] / self.grid_size[0],
                rel_pos[1] / self.grid_size[1],
                rel_vel[0] / self.max_speed,
                rel_vel[1] / self.max_speed,
                dist / np.linalg.norm(self.grid_size),
                np.cos(other.orientation - robot.orientation),
                np.sin(other.orientation - robot.orientation),
                1.0 if other.is_carrying else 0.0
            ])
        
        return obs
    
    def _calculate_rewards(self) -> Dict[int, float]:
        """Calculate rewards for each robot.

        Two modes:
        - sparse_rewards=False (default): Dense shaping for easier learning
        - sparse_rewards=True: Sparse rewards for meaningful evaluation

        Dense mode guides learning but may hide coordination benefits.
        Sparse mode is harder but better shows if MERA helps coordination.
        """
        rewards = {}

        for robot in self.robots:
            reward = 0.0

            # === DELIVERY REWARD (main objective) - always applied ===
            # Use _just_delivered tracking since carrying_package is cleared on delivery
            if robot.id in self._just_delivered:
                reward += self._just_delivered[robot.id]  # +100/200/300

            # === PICKUP REWARD (critical milestone) - always applied ===
            if robot.carrying_package is not None:
                pkg = self._get_package(robot.carrying_package)
                if pkg and pkg.pickup_time is not None:
                    time_since_pickup = self.current_step * self.dt - pkg.pickup_time
                    if time_since_pickup < 0.2:  # just picked up
                        reward += 25.0  # INCREASED from 10.0 to 25.0

            # === COLLISION PENALTY - always applied ===
            if robot.collision_count > 0:
                # Sparse mode: higher penalty to force coordination
                penalty = 50.0 if self.sparse_rewards else 2.0
                reward -= penalty * robot.collision_count
                robot.collision_count = 0

            # === DENSE SHAPING (only in dense mode) ===
            if not self.sparse_rewards:
                if robot.carrying_package is not None:
                    # Carrying package: reward for moving toward destination
                    pkg = self._get_package(robot.carrying_package)
                    if pkg and not pkg.is_delivered:
                        dist_to_dest = np.linalg.norm(robot.position - pkg.destination)
                        max_dist = np.linalg.norm(self.grid_size)  # ~70m diagonal
                        # DELIVERY FIX: Increased from 0.1 to 0.5 for stronger delivery signal
                        # Robots were picking up but not delivering because 0.1 was too weak
                        progress_reward = 0.5 * (1.0 - dist_to_dest / max_dist)
                        reward += max(0, progress_reward)
                else:
                    # Not carrying: reward for approaching packages
                    nearest_pkg_dist = float('inf')
                    for pkg in self.packages:
                        if not pkg.is_delivered and pkg.assigned_robot is None:
                            dist = np.linalg.norm(robot.position - pkg.position)
                            if dist < nearest_pkg_dist:
                                nearest_pkg_dist = dist

                    if nearest_pkg_dist < float('inf'):
                        max_dist = np.linalg.norm(self.grid_size)
                        # NAVIGATION FIX: Increased from 0.05 to 0.3 for stronger gradient
                        # With scattered packages, robots need clear signal to approach
                        progress_reward = 0.3 * (1.0 - min(nearest_pkg_dist, max_dist) / max_dist)
                        reward += progress_reward

                        # PRE-PICKUP REWARD: Teaches agent to slow down near packages
                        # Zone: 5m radius, reward scales with closeness and slowness
                        # Agent must learn: speed < 1.0 AND dist < 1.5 for actual pickup
                        if nearest_pkg_dist < 5.0:
                            # Speed factor: rewards slowing down (pickup requires speed < 1.0)
                            speed_factor = max(0, 1.0 - robot.speed)  # Full reward at speed=0
                            close_factor = 1.0 - nearest_pkg_dist / 5.0
                            pre_pickup_reward = 20.0 * speed_factor * close_factor
                            reward += pre_pickup_reward

                # Low battery penalty
                if robot.battery < 0.2 * self.battery_capacity:
                    reward -= 0.5

            rewards[robot.id] = reward
            self.episode_reward += reward

        return rewards
    
    def _get_package(self, package_id: int) -> Optional[Package]:
        """Get package by ID"""
        for pkg in self.packages:
            if pkg.id == package_id:
                return pkg
        return None
    
    def _get_info(self, robot: Robot) -> dict:
        """Get info dict for robot"""
        return {
            'position': robot.position.tolist(),
            'battery': robot.battery,
            'packages_delivered': robot.packages_delivered,
            'distance_traveled': robot.distance_traveled,
            'energy_consumed': robot.energy_consumed,
            'carrying': robot.is_carrying
        }
    
    def render(self, mode='human'):
        """Render environment (placeholder)"""
        # TODO: Implement visualization
        pass
    
    def close(self):
        """Cleanup"""
        pass
    
    def get_statistics(self) -> dict:
        """Get environment statistics"""
        return {
            **self.stats,
            'throughput': self.stats['packages_delivered'] / (self.current_step * self.dt + 1e-6) * 3600,
            'avg_waiting_time': self.stats['total_waiting_time'] / (self.stats['packages_delivered'] + 1e-6),
            'avg_distance': self.stats['total_distance'] / (self.stats['packages_delivered'] + 1e-6),
            'efficiency': self.stats['packages_delivered'] / (self.stats['total_energy'] + 1e-6)
        }
