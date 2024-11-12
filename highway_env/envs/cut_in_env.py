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

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                # Config

                "duration": 20,  # [s]
                "normalize_reward": True,
                # Rewards
                "collision_reward": -1, # Don't want to collide with the Cut-In Vehicle
                "high_speed_reward": 0.2, # Reward is minimal
                "reward_speed_range": [70, 80], #We want to keep pretty high speed
                # Ego Vehicle Setup
                "ego_lane_max_speed": 40, # m/s
                "ego_target_speed": 40, # m/s
                "ego_starting_speed": 40, # m/s
                # Cut In Vehicle Setup
                "other_vehicles_type": "highway_env.vehicle.behavior.CutInVehicle",
                "lane_max_speed": 30, # m/s
                "target_speed": 30, # m/s
                "starting_speed": 30, # m/s
                "min_distance_to_cut_in": 30, # This should likely be a calculated field based on the location and top speed
                "min_cut_in_start": 6,
                "max_cut_in_start": 21,
                # Obstacle Setup
                "min_obstacle_start": 350,
                "max_obstacle_start": 500
            }
        )
        return cfg

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
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1],)

        return reward

    def _rewards(self, action: Action) -> dict[str, float]:

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

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 550)

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

        net.add_lane(
            "a",
            "b",
            StraightLane(
                start=np.array([0, 0]),
                end=np.array([650, 0]),
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
                speed_limit=self.config["ego_lane_max_speed"]
            )
        )

        # merge lane
        merge_lane = StraightLane(
                start=np.array([0, StraightLane.DEFAULT_WIDTH]),
                end=np.array([650, StraightLane.DEFAULT_WIDTH]),
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
                forbidden=True,
                speed_limit=self.config["lane_max_speed"]
        )

        # Line to hold potential Cut-In vehicles
        net.add_lane(
            "a",
            "b",
            merge_lane
        )
        
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        # Force a Cut-In Scenario
        road.objects.append(Obstacle(road, merge_lane.position(np.random.randint(self.config["min_obstacle_start"], self.config["max_obstacle_start"]), 0)))

        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 0)).position(0, 0), speed=self.config["ego_starting_speed"]
        )
        ego_vehicle.target_speed = self.config["ego_target_speed"]
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("a", "b", 1)).position(np.random.randint(self.config["min_cut_in_start"], self.config["max_cut_in_start"]), 0), speed=self.config["starting_speed"]
        )
        merging_v.cut_before_obstacle_distance = self.config["min_distance_to_cut_in"]
        road.vehicles.append(merging_v)

