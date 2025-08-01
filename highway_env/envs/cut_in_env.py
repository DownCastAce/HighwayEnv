from __future__ import annotations

import numpy as np
import math
from typing import TypeVar, Optional

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle

Observation = TypeVar("Observation")

class CutInEnv(AbstractEnv):
    """
    A highway Cut-in negotiation environment.

    The ego-vehicle is driving on a highway and forced to cut-in due to an obstacle on the road, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "reward_version": 3,
                # Config
                "duration": 70,  # [s] is that max it would take going at 30 m/s to reach 2000m
                "normalize_reward": True,
                "lane_length": 2000,
                # Rewards
                "collision_reward": -3, # Don't want to collide with the Cut-In Vehicle
                "high_speed_reward": 0.7, # Reward is minimal
                "acceleration_reward": 0.1,
                "time_to_collision_reward": 0.7,
                "reward_speed_range": [70, 80], #We want to keep pretty high speed
                "reward_acceleration_range": [-2.5, 2.5],
                # Ego Vehicle Setup
                "ego_lane_max_speed": 40, # m/s
                "ego_target_speed": 40, # m/s
                "ego_starting_speed": 40, # m/s
                # Cut In Vehicle Setup
                "cut_in_buffer": 60, # m
                "other_vehicles_type": "highway_env.vehicle.behavior.CutInVehicle",
                "lane_max_speed": 30, # m/s
                "target_speed": 30, # m/s
                "starting_speed": [19.44, 25.0], # m/s [70, 90] km/h
                "min_distance_to_cut_in": 30, # This should likely be a calculated field based on the location and top speed
                # Obstacle Setup
                "obstacle_start": [200, 1800]
            }
        )
        return cfg

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        """
        Return a dictionary of additional information
        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = super()._info(obs, action)
        info["ego_vehicle_info"] = self._ego_vehicle_info()
        info["cut_in_vehicle_info"] = self._cut_in_vehicle_info()
        info["time_to_collision"] = self._time_to_collision()

        # Assume only one obstacle
        info["obstacle_position"] = float(self.road.objects[0].position[0])

        return info

    def _reward(self, action: Action) -> float:
        reward_version = self.config["reward_version"]

        if reward_version == 1:
            return self._reward_version_1(action)
        elif reward_version == 2:
            return self._reward_version_2(action)
        elif reward_version == 3:
            return self._reward_version_3(action)
        else:
            raise ValueError(f"Unknown reward version: {reward_version}")

    def _reward_version_1(self, action: Action) -> float:
        """
            The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
            :param action: the last action performed
            :return: the corresponding reward
        """
        rewards = self._rewards_version_1(action)

        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())

        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1], )

        return reward

    def _reward_version_2(self, action: Action) -> float:
        """
                The reward is defined to foster driving at high speed while avoiding collisions.
                """
        rewards = self._rewards_version_2(action)

        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())

        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] +
                                 self.config["acceleration_reward"] +
                                 self.config["time_to_collision_reward"]],
                                [0, 1])
        return reward

    def _reward_version_3(self, action: Action) -> float:
        """
                Composite reward: maintain max speed unless a car is ahead, then adapt reward
                to emphasize safety/distance and speed matching.
                """
        rewards = self._rewards_version_3(action)
        raw_reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )

        if self.config["normalize_reward"]:
            low = self.config["collision_reward"] * 1.1
            high = (
                    self.config["high_speed_reward"]
                    + self.config["acceleration_reward"]
                    + self.config["time_to_collision_reward"]
                    + self.config.get("max_speed_bonus", 0)
            )
            reward = utils.lmap(raw_reward, [low, high], [0, 1])
        else:
            reward = raw_reward

        return reward


    def _rewards(self, action: Action) -> dict[str, float]:
        reward_version = self.config["reward_version"]

        if reward_version == 1:
            return self._rewards_version_1(action)
        elif reward_version == 2:
            return self._rewards_version_2(action)
        elif reward_version == 3:
            return self._rewards_version_3(action)
        else:
            raise ValueError(f"Unknown reward version: {reward_version}")

    def _rewards_version_1(self, action: Action) -> dict[str, float]:
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        # Should we use Distance to vehicle infront instead of Crashed (>5?)
        # High speed
        # Maybe deceleration too much is unsafe? Wipelash or better not to crash
        # Or have crashed but certain velocity is determines the reward? Safe/Unsafe/Likely Death?
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1)
        }

    def _rewards_version_2(self, action: Action) -> dict[str, float]:
        # Calculate acceleration reward
        scaled_acceleration = utils.lmap(
            self.vehicle.action["acceleration"],
            self.config["reward_acceleration_range"],
            [0, 1]
        )

        acceleration_reward = float(np.clip(scaled_acceleration, 0, 1))

        # Calculate collision reward
        collision_reward = float(self.vehicle.crashed)

        # Calculate TTC reward
        ttc = self._time_to_collision()
        ttc_reward = self.ttc_reward_function(ttc)

        # Calculate forward speed
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)

        # Calculate speed reward
        max_speed = self.config["ego_lane_max_speed"]
        target_speed = self.config["ego_target_speed"]
        speed_reward = self.speed_reward_function(forward_speed, target_max=target_speed, max_speed=max_speed)

        return {
            "acceleration_reward": acceleration_reward,
            "collision_reward": collision_reward,
            "time_to_collision_reward": ttc_reward,
            "high_speed_reward": speed_reward
        }

    def _rewards_version_3(self, action: Action) -> dict[str, float]:
        """Compute reward components based on the traffic context w/2-flows approach."""
        # Compute TTC
        ttc, front_vehicle, distance = self._front_vehicle_info()

        # --- 1. No car ahead: maximize speed
        if ttc == float('inf'):
            speed_reward = self.speed_reward_function(self.vehicle.speed)
            safety_reward = 1.0
            match_speed_reward = 0.0  # Not relevant

        # --- 2. Car ahead: modulate rewards
        else:
            # How close are we? Use a reasonable threshold, e.g. safe_follow_distance
            safe_distance = max(self.vehicle.speed * 2.0, 10.0)  # eg. 2s rule
            # If close: reward safe following and matching front car's speed
            if distance < safe_distance * 1.2:
                # Reward matching speed with car ahead (velocity difference small)
                relative_speed = self.vehicle.speed - front_vehicle.speed
                match_speed_reward = math.exp(-abs(relative_speed) / 2.0)
                # Safety reward based on distance
                # (penalize being too close)
                distance_reward = math.exp(-abs(distance - safe_distance) / 5.0)
                safety_reward = 0.5 * self.ttc_reward_function(ttc) + 0.5 * distance_reward
                # Encourage staying just under or at the front vehicle's speed
                speed_reward = 0.3 * self.speed_reward_function(self.vehicle.speed) + \
                               0.7 * match_speed_reward
            else:
                # Not close yet, treat much like "no car ahead" but with ttc awareness
                speed_reward = self.speed_reward_function(self.vehicle.speed)
                safety_reward = self.ttc_reward_function(ttc)
                match_speed_reward = 0.0

        scaled_acceleration = utils.lmap(
            self.vehicle.action["acceleration"],
            self.config["reward_acceleration_range"],
            [0, 1]
        )
        acceleration_reward = float(np.clip(scaled_acceleration, 0, 1))
        collision_reward = float(self.vehicle.crashed)

        return {
            "high_speed_reward": speed_reward,
            "acceleration_reward": acceleration_reward,
            "time_to_collision_reward": safety_reward,
            "match_speed_reward": match_speed_reward,
            "collision_reward": collision_reward
        }

    def ttc_reward_function(self, ttc):
        """
        Calculate Time to Collision reward based on the Time to Collision

        ttc: Time before a collision will occur
        """
        if ttc == float('inf'):
            return 1.0  # Maximum reward for no collision risk
        elif ttc <= 0:
            return 0.0  # Minimum reward for imminent collision
        else:
            # Exponential decay function
            return 1 - math.exp(-ttc / 3)

    def speed_reward_function(self, speed, min_speed=22.22, target_min=26.39,
                              target_max=29.17, max_speed=33.33):
        if speed < 0:
            return -2  # Stronger penalty for reversing!
        if speed < min_speed:
            return -1
        elif min_speed <= speed < target_min:
            return np.interp(speed, [min_speed, target_min], [0.1, 0.7])
        elif target_min <= speed <= target_max:
            return 1
        elif target_max < speed <= max_speed:
            return np.interp(speed, [target_max, max_speed], [1, -0.5])
        else:
            return -1

    def _time_to_collision(self) -> float:
        """Return the time-to-collision to the front vehicle in current lane, or inf if none."""
        ttc, _, _ = self._front_vehicle_info()
        return ttc

    def _front_vehicle_info(self) -> tuple[float, Optional[Vehicle], float]:
        """Return (ttc, front_vehicle, distance) for the closest vehicle ahead."""
        ego_vehicle = self.vehicle
        ego_lane = self.vehicle.lane_index
        vehicles_ahead = [
            (v, v.position[0] - ego_vehicle.position[0])
            for v in self.road.vehicles
            if v.lane_index == ego_lane and v != ego_vehicle and v.position[0] > ego_vehicle.position[0]
        ]
        if not vehicles_ahead:
            return float('inf'), None, float('inf')
        front_vehicle, distance = min(vehicles_ahead, key=lambda x: x[1])
        distance -= getattr(front_vehicle, 'DISTANCE_WANTED', 0.0)
        ego_v = ego_vehicle.speed * np.cos(ego_vehicle.heading)
        front_v = front_vehicle.speed * np.cos(front_vehicle.heading)
        relative_speed = ego_v - front_v

        if relative_speed <= 0:
            ttc = float('inf')
        else:
            ttc = distance / relative_speed
        ttc -= getattr(front_vehicle, 'TIME_WANTED', 1.5)
        ttc = max(0, ttc)
        return ttc, front_vehicle, distance

    def _ego_vehicle_info(self) -> dict:
        return {
            "position": float(self.vehicle.position[0]),
            "lane": self.vehicle.lane_index[2],
            "speed": self.vehicle.speed,
            "acceleration": self.vehicle.action["acceleration"]
        }

    def _cut_in_vehicle_info(self) -> dict:
        # grab cut-in vehicle, assume only one
        cut_in_vehicle = next(filter(lambda v: v != self.vehicle, self.road.vehicles), None)
        if cut_in_vehicle is None:
            return {
                "position": None,
                "speed": None,
                "acceleration": None
            }
        return {
            "position": float(cut_in_vehicle.position[0]),
            "lane": cut_in_vehicle.lane_index[2],
            "speed": cut_in_vehicle.speed,
            "acceleration": cut_in_vehicle.action["acceleration"]
        }

    def _vehicle_positions(self) -> dict[str, float]:
        """Determines the Time to Collision (TTC)"""
        ego_vehicle = self.vehicle
        ego_lane = self.vehicle.lane_index
        vehicle_in_front = next(filter(lambda v: v.lane_index == ego_lane and v != ego_vehicle, self.road.vehicles),
                                None)

        # If there are none then the TTC is infinity
        if vehicle_in_front is None:
            return { "ego_vehicle_position": ego_vehicle.position[0], "vehicle_in_front_position": None, "distance": float("inf") }

        # Distance between the ego and vehicle in front (If any)
        distance = vehicle_in_front.position[0] - ego_vehicle.position[0]

        return { "ego_vehicle_position": ego_vehicle.position[0], "vehicle_in_front_position": vehicle_in_front.position[0], "distance": distance }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > self.config["lane_length"])

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        lane_max_length = self.config["lane_length"]

        net.add_lane(
            "a",
            "b",
            StraightLane(
                start=np.array([0, 0]),
                end=np.array([lane_max_length, 0]),
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
                speed_limit=self.config["ego_lane_max_speed"]
            )
        )

        # cut-in lane
        cut_in_lane = StraightLane(
                start=np.array([0, StraightLane.DEFAULT_WIDTH]),
                end=np.array([lane_max_length, StraightLane.DEFAULT_WIDTH]),
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
                forbidden=True,
                speed_limit=self.config["lane_max_speed"]
        )

        # Line to hold potential Cut-In vehicles
        net.add_lane("a","b", cut_in_lane)
        
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        # Range
        low, high = self.config["obstacle_start"]
        obstacle_x = np.random.randint(low, high)
        obstacle_point = cut_in_lane.position(obstacle_x, 0)

        # Force a Cut-In Scenario
        road.objects.append(Obstacle(road, obstacle_point))

        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road

        ego_velocity = self.config["ego_starting_speed"]
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 0)).position(0, 0), speed=ego_velocity
        )
        ego_vehicle.target_speed = self.config["ego_target_speed"]
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        cut_in_start_speed_range = self.config["starting_speed"]
        cut_in_velocity = np.random.randint(cut_in_start_speed_range[0], cut_in_start_speed_range[1])
        obstacle_x, _ = road.objects[0].position

        cut_in_accel = self.acceleration(cut_in_velocity, self.config["lane_max_speed"])
        cut_in_start = self.calc_cut_in_start(ego_velocity, cut_in_velocity, self.config["lane_max_speed"], cut_in_accel, obstacle_x, self.config["cut_in_buffer"] )

        cut_in_v = other_vehicles_type(
            road, road.network.get_lane(("a", "b", 1)).position(cut_in_start, 0), speed=cut_in_velocity
        )
        cut_in_v.target_speed = self.config["target_speed"]
        cut_in_v.cut_before_obstacle_distance = self.config["min_distance_to_cut_in"]
        road.vehicles.append(cut_in_v)

    def acceleration(self, speed, target_speed) -> float:
        """"""
        return 3.0 * (1 - np.power(max(speed, 0) / abs(utils.not_zero(target_speed)), self.road.np_random.uniform(low=3.5, high=4.5)))

    def calc_cut_in_start(self,
                          ego_speed: float,
                          cutin_speed_init: float,
                          cutin_speed_max: float,
                          cutin_accel: float,
                          obstacle_pos: float,
                          merge_buffer: float
                          ) -> float:
        """
        Calculate the starting position for the cut-in vehicle.

        :param ego_speed: Ego vehicle speed (m/s)
        :param cutin_speed_init: Cut-in vehicle initial speed (m/s)
        :param cutin_speed_max: Cut-in vehicle maximum speed (m/s)
        :param cutin_accel: Cut-in vehicle acceleration (m/s^2)
        :param obstacle_pos: Obstacle position along the lane (m)
        :param merge_buffer: Desired buffer distance at the cut-in point (m)
        :return: Starting position for the cut-in vehicle (m)
        """
        # Distance from ego to obstacle minus the safety buffer
        x_ego_to_obstacle = obstacle_pos - merge_buffer

        # Time until ego reaches the buffer point
        t_until_buffer = x_ego_to_obstacle / ego_speed

        # Time for the cut-in vehicle to accelerate from initial-to-max speed
        t_accel = (cutin_speed_max - cutin_speed_init) / cutin_accel

        # Time spent at max speed before reaching the buffer point
        t_at_max = max(0.0, t_until_buffer - t_accel)

        # Distance covered during acceleration
        d_accel = (cutin_speed_init * t_accel + 0.5 * cutin_accel * t_accel ** 2)

        # Total distance the cut-in vehicle travels before merge
        d_total = d_accel + cutin_speed_max * t_at_max

        # Compute the starting position so that it arrives at the buffer point just in time
        start_pos = obstacle_pos - d_total

        # Ensure non-negative start position
        return max(0.0, start_pos)

