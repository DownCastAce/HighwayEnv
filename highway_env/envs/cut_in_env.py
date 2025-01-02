from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle

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
                # Config
                "duration": 60,  # [s]
                "normalize_reward": True,
                "lane_length": 2000,
                # Rewards
                "collision_reward": -1, # Don't want to collide with the Cut-In Vehicle
                "high_speed_reward": 0.2, # Reward is minimal
                "acceleration_reward": 0.5,
                "reward_speed_range": [70, 80], #We want to keep pretty high speed
                "reward_acceleration_range": [-2.5, 1.5],
                "time_to_collision_reward": 0.5,
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

    def _info(self, obs, action) -> dict:
        """
                Return a dictionary of additional information

                :param obs: current observation
                :param action: current action
                :return: info dict
                """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            # distance to car infront
            # v_c, 
            # "ttc": self._time_to_collision(),
            "action": action,
            "acceleration": self.vehicle.action["acceleration"]

        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    # "acceleration"
    # ToDo : Need to specify the new rewards for this
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)

        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())

        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"] + self.config["acceleration_reward"], self.config["high_speed_reward"]], [0, 1],)

        return reward

    def _rewards(self, action: Action) -> dict[str, float]:

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        scaled_acceleration = utils.lmap(
            self.vehicle.action["acceleration"], self.config["reward_acceleration_range"], [0, 1]
        )

        ttc = self._time_to_collision()
        if ttc == float('inf') or ttc > 2:
            ttc = 1.0
        elif ttc < 0:
            ttc = 0.25  # Less than half for a negative number
        elif ttc < 2:
            ttc = 0.0  # Minimum for < 2
        else:
            ttc = (ttc - 0) / (2 - 0)  # Linear interpolation between 0 and 2

        # Should we use Distance to vehicle infront instead of Crashed (>5?)
        # High speed
        # Maybe deceleration too much is unsafe? Wipelash or better not to crash
        # Or have crashed but certain velocity is determines the reward? Safe/Unsafe/Likely Death?
        return {
            "acceleration_reward": np.clip(scaled_acceleration, 0, 1),
            "collision_reward": float(self.vehicle.crashed),
            "time_to_collision_reward": float(ttc),
            "high_speed_reward": np.clip(scaled_speed, 0, 1)
        }

    def _time_to_collision(self) -> float:
        """Determines the Time to Collision (TTC)"""
        ego_vehicle = self.vehicle
        ego_lane = self.vehicle.lane_index
        vehicle_in_front = next(filter(lambda v: v.lane_index == ego_lane and v != ego_vehicle, self.road.vehicles), None)

        # If there are none then the TTC is infinity
        if vehicle_in_front is None:
            return float("inf")

        # Distance between the ego and vehicle in front (If any)
        distance = vehicle_in_front.position[0] - ego_vehicle.position[0]

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        ego_v = ego_vehicle.speed * np.cos(ego_vehicle.heading)

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        vehicle_in_front_v = vehicle_in_front.speed * np.cos(vehicle_in_front.heading)

        # TTC is calculated via Distance and the Relative speed difference between the cars
        return distance / (ego_v - vehicle_in_front_v)

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
        net.add_lane(
            "a",
            "b",
            cut_in_lane
        )
        
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        # Range
        obstacle_location_range = self.config["obstacle_start"]
        obstacle_start = np.random.randint(obstacle_location_range[0], obstacle_location_range[1])

        # Force a Cut-In Scenario
        road.objects.append(Obstacle(road, cut_in_lane.position(obstacle_start, 0)))

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
        cut_in_v.target_speed = self.config["ego_target_speed"]
        cut_in_v.cut_before_obstacle_distance = self.config["min_distance_to_cut_in"]
        road.vehicles.append(cut_in_v)

    def acceleration(self, speed, target_speed) -> float:
        """"""
        return 3.0 * (1 - np.power(max(speed, 0) / abs(utils.not_zero(target_speed)), self.road.np_random.uniform(low=3.5, high=4.5)))

    def calc_cut_in_start(self, v_e, v_c, m_v_c, a_c, x_o, buffer) -> float:
        """"""
        x_eo = x_o - buffer
        t_x_eo = x_eo / v_e

        # Time to reach max velocity
        t_r_m_v_c = abs(m_v_c - v_c) / a_c

        # Distance travelled during accelerations
        # Assuming Constant Acceleration
        a_x_c = v_c * t_r_m_v_c + (0.5 * a_c * t_r_m_v_c**2)

        # Time at max velocity
        t_m_v_c = t_x_eo - t_r_m_v_c

        x_c = x_o - (m_v_c * t_m_v_c) - a_x_c

        return x_c

