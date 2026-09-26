"""Generate object viewpoints by tracing boundaries, sampling radial positions,
filtering for navigability and visibility, and orienting valid points toward the object.
"""

from itertools import product as iter_prod
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.simulator import Simulator

from habitat.utils.visualizations import maps

from typing import Dict, Union, List

TEST_SCENE_ID = "106366410_174226806"  # "102816756" 
TEST_SCENE_DATASET_PATH = "/mnt/vlfm_query_embed/habitat-lab/data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"


class ViewPoints_Generator:
    """Find boundary candidates, sample nearby concentric viewpoints, and validate visibility.
    Visibility validation uses projected geometry and depth, not semantic masks.
    """

    def __init__(
        self,
        sim: Simulator = None,
        scene_id: str = TEST_SCENE_ID,
        scene_dataset_path: str = TEST_SCENE_DATASET_PATH,
        override_sim_settings: Dict = {},
    ):

        self.scene_id = scene_id
        self.sim = (
            sim
            if sim is not None
            else self.make_sim(scene_id, scene_dataset_path, override_sim_settings)
        )

        self.sensor_h, self.sensor_w = (
            self.sim.config.agents[0].sensor_specifications[0].resolution
        )
        self.sensor_vert_pos = (
            self.sim.config.agents[0].sensor_specifications[0].position[1]
        )
        self.hfov = self.sim.config.agents[0].sensor_specifications[0].hfov.__float__()

        self.vfov_rad = self.get_vfov()
        self.focus = self.sensor_w / (2 * np.tan(np.deg2rad(self.hfov) / 2))

        self.view_pt_valid_thresh = 0.1

    def make_sim(
        self, scene_id: str, scene_dataset_path: str, override_sim_settings: Dict = {}
    ):

        sim_settings = {
            "scene": scene_id,
            "scene_dataset": scene_dataset_path,
            "default_agent": 0,  # Index of the default agent
            "sensor_vert_position": 0.88,  # Height of Sensor
            "agent_radius": 0.18,  # Radius of Agent
            "width": 640,  # Sensor Width
            "height": 480,  # Sensor Height
            "hfov": 79,
            "agent_max_climb": 0.2,
            "cell_height": 0.2,
            "enable_physics": False,
            "gpu_device_id": 0,
        }

        if len(override_sim_settings) > 0:
            for key in override_sim_settings:
                sim_settings[key] = override_sim_settings[key]

        # Build Simulator Configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = sim_settings["gpu_device_id"]
        sim_cfg.scene_id = sim_settings["scene"]
        sim_cfg.scene_dataset_config_file = sim_settings["scene_dataset"]
        sim_cfg.enable_physics = sim_settings["enable_physics"]

        # Note: all sensors must have the same resolution
        sensor_specs = []

        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
        color_sensor_spec.position = [0.0, sim_settings["sensor_vert_position"], 0.0]
        color_sensor_spec.hfov = sim_settings["hfov"]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
        depth_sensor_spec.position = [0.0, sim_settings["sensor_vert_position"], 0.0]
        depth_sensor_spec.hfov = sim_settings["hfov"]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

        # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        # semantic_sensor_spec.uuid = "semantic_sensor"
        # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        # semantic_sensor_spec.resolution = [sim_settings["height"], sim_settings["width"]]
        # semantic_sensor_spec.position = [0.0, sim_settings["sensor_vert_position"], 0.0]
        # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        # sensor_specs.append(semantic_sensor_spec)

        # Build Agent Configuration: Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        # Build Navmesh
        print(f"Pathfinder is Loaded: {sim.pathfinder.is_loaded}")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = sim_settings["sensor_vert_position"]
        navmesh_settings.agent_radius = sim_settings["agent_radius"]
        navmesh_settings.agent_max_climb = sim_settings["agent_max_climb"]
        navmesh_settings.cell_height = sim_settings["cell_height"]
        navmesh_settings.include_static_objects = True
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        print(f"Navmesh Recompute Success: {navmesh_success}")

        return sim

    ## Generate Boundary Points
    def is_boundary_pt(self, ref_point: np.ndarray, _delta=0.05) -> bool:

        pt = ref_point.copy()
        pt[1] = self._get_floor_height(pt)

        # Perturb Point in eight different directions
        pt_perturb_cross = [
            self.perturb_along_dim(pt.copy(), dim, i * _delta)
            for dim in [0, 2]
            for i in [1, -1]
        ]
        pt_perturb_diag = [
            self.perturb_along_dim(pt.copy(), [0, 2], i * np.array(_delta))
            for i in iter_prod([1, -1], repeat=2)
        ]
        pt_perturb = np.concatenate((pt_perturb_cross, pt_perturb_diag), axis=0)

        is_not_navigable = [
            not self.sim.pathfinder.is_navigable(point) for point in pt_perturb
        ]
        if np.all(is_not_navigable):
            return False  # Non Navigable Point
        if np.any(is_not_navigable):
            return True  # Boundary Point
        return False  # Navigable Point

    @staticmethod
    def perturb_along_dim(
        pt: Union[np.ndarray, List], dim: Union[int, List], delta: Union[float, List]
    ):
        r"""
        Args:
            - pt: Point to perturb.
            - dim: Dimension to perturb along. Can be an integer of a list of integers.
            - delta: Amount to perturb along each dimension.

        """

        point = pt.copy()

        # Perturb only along one dimension
        if isinstance(dim, int):
            point[dim] += delta
            return point

        # Perturb along one or more dimensions
        assert len(delta) == len(
            dim
        ), "Length of Delta values should match length of dimensions to perturb."
        for curr_dim, curr_delta in zip(dim, delta):
            point[curr_dim] += curr_delta

        return point

    def shoot_pts_towards(
        self,
        start_pos: np.ndarray,
        degrees: float,
        shoot_till: float,
        resolution_fact: int = 1,
        _num_samples: int = 10,
    ):
        r"""
        Args:
            - start_pos: Starting position from which to shoot points
            - degrees: Orientation to face towards when shooting points
            - resolution_fact: Resolution Factor, results in smaller step size
            - num_samples: Number of evenly-spaced points to shoot from the start position
        """

        assert resolution_fact > 0

        # Step size from the start position to the final limit (shoot_till)
        _delta = shoot_till / (_num_samples * resolution_fact)

        # Direction to step towards
        rad = np.deg2rad(degrees)
        dx_unit, dz_unit = np.cos(rad), np.sin(rad)

        # Shoot evenly-spaced points towards the required direction
        shoot_pts = [
            self.perturb_along_dim(
                start_pos.copy(), [0, 2], i * _delta * np.array([dx_unit, dz_unit])
            )
            for i in range(_num_samples)
        ]

        return shoot_pts

    def shoot_point_to_boundary(
        self, obj_pos: np.ndarray, degrees: float, _shoot_till: float = 2
    ):
        r"""
        Args:
            - sim: Simulator
            - obj_pos: Object Position (should be a non-navigable point)
            - degrees: Angle (in degrees) to face at, measured with respect to the object position
            - _shoot_till: Generate points till this limit (in meters)
            - _max_tries: Maximum number of trials to find the boundary point
        Returns:
            - found_boundary_pt (bool)
            - Boundary Point
        """

        # Ground the point to the floor height
        pt = obj_pos.copy()
        pt[1] = self._get_floor_height(pt)
        assert not self.sim.pathfinder.is_navigable(
            pt
        ), "Please provide a non-navigable position as obj_pos"

        # Generate points along the angle direction till the given limit (shoot_till)
        # Check navigability of points. If none are navigable, return
        shoot_pts = self.shoot_pts_towards(
            start_pos=pt.copy(), degrees=degrees, shoot_till=_shoot_till
        )
        is_nav_pts = [self.sim.pathfinder.is_navigable(point) for point in shoot_pts]
        if np.all(np.invert(is_nav_pts)):
            return False, None

        # Get the transition points: One point is non-navigable while the other is navigable
        ind_transition = np.argwhere(is_nav_pts)[0].item()
        low_bound_pt = pt if ind_transition == 0 else shoot_pts[ind_transition - 1]

        # Snap point
        return True, self.sim.pathfinder.snap_point(low_bound_pt)

    def next_valid_boundary_pt(
        self,
        obj_pos: np.ndarray,
        _shoot_till: float = 2,
        start_degree: float = 0,
        delta_degrees: float = 5,
    ):
        r"""
        Varies the degree starting from start_degree, and returns when we reach the first boundary point

        Args:
            - sim: Simulator Object
            - obj_pos: Object Position. Used to find the boundary with reference to this point.
            - _shoot_till: Distance to check for the boundary.
            - start_degree: Starting degree value to search for boundary point from.
            - delta_degrees: Search step to vary degrees by.
        """
        degrees = start_degree - delta_degrees
        while degrees <= (360 + start_degree):
            degrees += delta_degrees
            found_pt, pt = self.shoot_point_to_boundary(obj_pos, degrees, _shoot_till)
            if found_pt:
                return pt, degrees

        return None, degrees

    def next_boundary_pt_at_dist(
        self,
        obj_pos: np.ndarray,
        ref_pt: np.ndarray,
        pts_dist: float,
        curr_degree: float,
        delta_degrees: float,
        start_degree=0,
        _shoot_till: float = 2,
        invalid_ranges: float = [],
    ):
        r"""
        Binary Search for the next boundary pt at a specific distance to ref_pt.

        Args:
            - sim: Simulator object
            - obj_pos: Object Position.
            - ref_pt: Reference point from which the next boundary point is to be found.
            - pts_dist: Distance between the reference point and the next boundary point.
            - curr_degree: Current angle value in degrees, corresponding to the reference point.
            - delta_degrees: Step size for varying degrees.
            - start_degree: Upper limit offset for degrees.
            - _shoot_till: Distance to check for the boundary. (Look into shoot_point_to_boundary)
            - invalid_ranges: Invalid ranges of degree values, denoting non-boundary regions.

        """

        search_depth = 1
        skip_iter = False
        curr_degree += delta_degrees / 2

        diff_resolution = 0.05
        max_search_depth = math.ceil(np.log2(1 / diff_resolution))

        # Searches a full 360 to find the next point at specific distance
        while curr_degree < (360 + start_degree):

            # Find potential boundary point at current degree value
            found_pt, potential_pt = self.shoot_point_to_boundary(
                obj_pos, curr_degree, _shoot_till
            )
            if not found_pt:

                # If current degree is in an invalid range (region without a boundary),
                # then update current degree to end of invalid range
                for invalid_range in invalid_ranges:
                    lower_lim, upper_lim = invalid_range[0], invalid_range[1]

                    if lower_lim > upper_lim:
                        if (upper_lim < curr_degree < 360) or (
                            0 < curr_degree < lower_lim
                        ):
                            curr_degree = lower_lim
                            search_depth += 1
                            skip_iter = True

                    else:
                        if lower_lim < curr_degree < upper_lim:
                            curr_degree = invalid_range[1]
                            search_depth += 1
                            skip_iter = True

                if skip_iter:
                    skip_iter = False
                    continue

                # If no invalid range is found, then this region is a first occurrence.
                # Scan through degrees and get the first boundary point. Update the invalid ranges.
                invalid_degree_start = curr_degree
                potential_pt, curr_degree = self.next_valid_boundary_pt(
                    obj_pos, _shoot_till, start_degree=curr_degree, delta_degrees=1
                )
                invalid_ranges.append((invalid_degree_start, curr_degree))

                # return potential_pt, curr_degree+(diff_resolution*10), invalid_ranges
                return potential_pt, curr_degree, invalid_ranges

            # Get the distance between the potential point and the previous boundary point (ref_pt)
            dist_from_prev = self.dist_btw_pts(potential_pt[[0, 2]], ref_pt[[0, 2]])

            # print(f"Next potential bound point found. Distance from prev: {dist_from_prev}, Current Degree: {curr_degree}, Search Depth: {search_depth}")

            # Binary Search Implementation
            # If the search goes on for more than 20 steps, then we force a solution to solve the indecision
            if (abs(dist_from_prev - pts_dist) < diff_resolution) or (
                search_depth > max_search_depth
            ):
                return potential_pt, curr_degree, invalid_ranges

            elif dist_from_prev < pts_dist:
                if search_depth > 1:
                    search_depth += 1
                curr_degree += delta_degrees / (2**search_depth)

            elif dist_from_prev > pts_dist:
                search_depth += 1
                curr_degree -= delta_degrees / (2**search_depth)

        return None, curr_degree, invalid_ranges

    def boundary_around_nav_obj(
        self, obj_pos: np.ndarray, obj_dims: np.ndarray, pts_dist=0.5, with_center=True
    ):

        if obj_dims is None:
            print("obj dims is none")
            return [obj_pos]

        bound_pts = []
        obj_corners = []
        floor_height = self._get_floor_height(obj_pos.copy())

        # Get object corners
        delta_vals = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]])
        obj_corners = [
            self.perturb_along_dim(obj_pos, [0, 2], delta * obj_dims[[0, 2]])
            for delta in delta_vals
        ]

        # Generate points around the object dimensions
        for ind in range(len(obj_corners)):

            corner_prev = obj_corners[ind - 1]
            corner_next = obj_corners[ind]

            # Generate points in between the two corners
            dist = self.dist_btw_pts(corner_prev[[0, 2]], corner_next[[0, 2]])
            num_pts = math.ceil(dist / pts_dist)
            if num_pts > 2:
                pts_in_btw = self.generate_pts_btw(corner_prev, corner_next, num_pts)[
                    :-1
                ]
            else:
                pts_in_btw = [corner_prev]

            pts_in_btw = [
                pt
                for pt in pts_in_btw
                if self.sim.pathfinder.is_navigable([pt[0], floor_height, pt[2]])
            ]
            if len(pts_in_btw) == 0:
                continue

            pts_in_btw = np.array(pts_in_btw)

            # Ensure 2D
            if pts_in_btw.ndim == 1:
                pts_in_btw = pts_in_btw[np.newaxis, :]

            if len(bound_pts) == 0:
                bound_pts = pts_in_btw
            else:
                bound_pts = np.concatenate((bound_pts, pts_in_btw))

        if len(bound_pts) == 0:
            return []

        # Add center point (obj_pos) to boundary points
        if with_center:

            bound_pts = np.concatenate((obj_pos[np.newaxis, :], bound_pts))

        return bound_pts

    def boundary_around_obj(
        self,
        obj_pos: np.ndarray,
        obj_dims: np.ndarray = None,
        pts_dist: float = 0.5,
        delta_degrees: float = 20,
        keep_final_pt: bool = True,
        _shoot_till: float = 2,
    ):
        r"""
        Args:
            - sim: Simulator object.
            - obj_pos: Object Position
            - obj_dims: Object Dimensions
            - pts_dist: Distance between two boundary points
            - delta_degrees: Step size for varying degree values.
            - keep_final_pt: If False, removes the last boundary point if its
                            distance to the initial point is less than pts_dist
            - _shoot_till: Maximum distance to check for possible boundary point
        """

        pt = obj_pos.copy()
        pt[1] = self._get_floor_height(pt)
        # print(f"Object Pos: {obj_pos}, Floor Height: {pt[1]}")

        if self.sim.pathfinder.is_navigable(pt):
            print(f"Object position is in a navigable region!")
            return True, self.boundary_around_nav_obj(
                np.array(obj_pos), np.array(obj_dims), pts_dist, with_center=True
            )

        boundary_pts = []
        invalid_ranges = []
        start_degree = 0

        # Start Boundary Point, which will be used as reference for subsequent boundary points
        start_pt, start_degree = self.next_valid_boundary_pt(
            obj_pos, _shoot_till, start_degree=start_degree, delta_degrees=1
        )
        if start_pt is None:
            return False, []
        boundary_pts.append(start_pt)
        degrees = start_degree

        # Iterates through 360 degrees, making sure all possible boundary points are attained
        while degrees < (360 + start_degree):

            # print(f"\n\nDegree: {degrees}, bound_pts: {len(boundary_pts)}, invalid_ranges: {invalid_ranges}")

            # Next boundary point, with reference to the last sampled boundary point
            next_pt, degrees, invalid_ranges = self.next_boundary_pt_at_dist(
                obj_pos,
                ref_pt=boundary_pts[-1],
                pts_dist=pts_dist,
                curr_degree=degrees,
                start_degree=start_degree,
                delta_degrees=delta_degrees,
                _shoot_till=_shoot_till,
                invalid_ranges=invalid_ranges,
            )

            if next_pt is not None:
                boundary_pts.append(next_pt)

        if (not keep_final_pt) and (
            self.dist_btw_pts(boundary_pts[0][[0, 2]], boundary_pts[-1][[0, 2]])
            < pts_dist
        ):
            boundary_pts = boundary_pts[:-1]
        return True, boundary_pts

    ## Generate Radial Navigable Samples around the Boundary
    def sample_radial_nav_pts(
        self, ref_pt: np.ndarray, max_radius: float = 1, pts_dist: float = 0.2
    ):
        r"""
        Generate radial and navigable samples centered around a reference point.
        These are generated at a fixed radius step and a fixed angular step.

        Args:
            - ref_pt: Reference point, used as the central reference to generate samples
            - max_radius: Maximum radius to generate points till
            - pts_dist: Distance between the generated samples.

        """

        radial_pts = []
        radius_info = []
        step_radius = pts_dist

        for radius in np.arange(step_radius, max_radius + step_radius, step_radius):

            # Get angular step value
            perimeter = 2 * np.pi * radius
            num_concentric_pts = perimeter / pts_dist
            step_degree = math.ceil(360 / num_concentric_pts)

            for degree in np.arange(0, 360, step_degree):

                # Get the new sample point
                rad = np.deg2rad(degree)
                dx, dz = radius * np.cos(rad), radius * np.sin(rad)
                radial_pt = [ref_pt[0] + dx, ref_pt[1], ref_pt[2] + dz]

                # Check for navigability
                if self.sim.pathfinder.is_navigable(radial_pt):
                    radial_pts.append(radial_pt)
                    radius_info.append(radius)

        return np.array(radial_pts), np.array(radius_info)

    def samples_from_boundary(
        self,
        boundary_pts: np.ndarray,
        sample_max_radius: float = 1,
        pts_dist: float = 0.2,
    ):
        r"""
        Generate Navigable samples around the boundary.

        Args:
            - boundary_pts: Points lying around the boundary
            - pts_dist: Distance between the generated sample points.

        """

        # Keeps track of active boundary points, useful for reducing overlap during sampling
        active_boundary_pts = [True] * len(boundary_pts)
        ref_outer_pts = []
        sampled_pts = []

        while sum(active_boundary_pts) > 0:

            active_ind = active_boundary_pts.index(True)
            active_bound_pt = boundary_pts[active_ind]

            # Generate navigable radial samples around the current active boundary point
            radial_nav_samples, radius_info = self.sample_radial_nav_pts(
                active_bound_pt, max_radius=sample_max_radius, pts_dist=pts_dist
            )

            if len(radial_nav_samples) == 0:
                active_boundary_pts[active_ind] = False
                continue

            # If previously sampled, use the previous outer valid points to filter current points based on minimum distance
            # A new sample should be at least pts_dist away from all the previous outer points
            if len(ref_outer_pts) > 0:
                is_valid_mask = [True] * len(radial_nav_samples)
                for ind, pt in enumerate(radial_nav_samples):
                    is_valid = np.all(
                        [
                            self.dist_btw_pts(pt, ref_pt) > pts_dist
                            for ref_pt in ref_outer_pts
                        ]
                    )
                    is_valid_mask[ind] = is_valid

                radial_nav_samples = radial_nav_samples[is_valid_mask]
                radius_info = radius_info[is_valid_mask]

            # Append the valid sampled points
            if len(sampled_pts) == 0:
                sampled_pts = radial_nav_samples
            else:
                sampled_pts = np.concatenate((sampled_pts, radial_nav_samples), axis=0)

            # Update the outer points for reference
            ref_outer_pts = radial_nav_samples[radius_info == max(radius_info)]

            # Deactivate all boundary points within the range of the outer points
            active_boundary_pts[active_ind] = False
            if sum(active_boundary_pts) <= 0:
                break

            # Deactivate boundary Points before the current boundary point
            within_range = True
            left_ind = active_ind - 1
            while within_range and left_ind >= 0:

                within_range = np.any(
                    [
                        self.dist_btw_pts(ref_pt, boundary_pts[left_ind]) <= pts_dist
                        for ref_pt in ref_outer_pts
                    ]
                )

                if within_range:
                    active_boundary_pts[left_ind] = False
                left_ind -= 1
            if sum(active_boundary_pts) <= 0:
                break

            # Deactive boundary Points after the current boundary point
            within_range = True
            right_ind = active_ind + 1
            while within_range and (right_ind < len(boundary_pts)):

                within_range = np.any(
                    [
                        self.dist_btw_pts(ref_pt, boundary_pts[right_ind]) <= pts_dist
                        for ref_pt in ref_outer_pts
                    ]
                )

                if within_range:
                    active_boundary_pts[right_ind] = False
                right_ind += 1

        return sampled_pts

    ## Obtaining Viewpoints
    @staticmethod
    def group_angles(angles: List, step_size: float):
        r"""
        For a given list of angles, creates groups of angles differing only by step_size.

        """

        groups = []
        curr_group = [0, 0]

        # Checks if the current angle is <step_size> away from the previous angle
        # If so, the current group is updated. This is done till a greater jump is observed.
        for ind in range(len(angles[1:])):

            diff = (angles[ind + 1] - angles[ind]) % 360  # Difference of angles

            if diff <= step_size:
                curr_group[1] = ind + 1  # Update Current group
            else:
                if curr_group[0] != curr_group[1]:
                    groups.append(curr_group)  # Add current group

                curr_group = [ind + 1, ind + 1]  # Reinitialize Current group

        if curr_group[0] != curr_group[1]:
            groups.append(curr_group)

        if len(groups) == 0:
            return []

        # Check if the difference between first and last angles is valid
        # If so, a new group extending forward and backward is created
        diff = (angles[0] - angles[-1]) % 360

        if diff <= step_size:

            curr_group = [len(angles) - 1, 0]

            if groups[-1][1] == len(angles) - 1:
                curr_group[0] = groups[-1][0] - len(angles)
                del groups[-1]

            if (len(groups) > 1) and (groups[0][0] == 0):
                curr_group[1] = groups[0][1]
                del groups[0]

            groups.append(curr_group)

        # Return groups with angles
        grouped_angles = [[angles[group[0]], angles[group[1]]] for group in groups]
        return np.array(grouped_angles)

    def face_object(
        self, object_position: np.ndarray, point: np.ndarray, return_yaw: bool = False
    ):
        r"""
        Faces the agent (at point) towards the object position.
        """
        EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
        cam_normal = (object_position - point) + EPS_ARRAY
        cam_normal[1] = 0
        cam_normal = cam_normal / np.linalg.norm(cam_normal)
        q = utils.quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)
        quat_coeffs = utils.quat_to_coeffs(q)

        if return_yaw:
            return self.quat_coeffs_to_yaw(quat_coeffs)
        return utils.quat_to_coeffs(q)

    def obs_at_pose(
        self,
        pos: np.ndarray,
        rot: float = None,
    ):
        r"""
        Get the RGB and Depth observations at position.
        """

        agent_state = habitat_sim.AgentState()
        agent_state.position = pos
        if rot is not None:
            agent_state.rotation = rot

        self.sim.agents[0].set_state(agent_state, infer_sensor_states=False)
        return self.sim.get_sensor_observations()

    def is_obj_in_frame(self, obj_pos: np.ndarray, agent_pos: np.ndarray):
        r"""
        Checks if object height is viewable from agent's perspective.
        """

        # Half of VFOV
        half_vfov_rad = self.vfov_rad / 2

        # Depth value: Distance from agent to object
        depth_val = self.dist_btw_pts(obj_pos[[0, 2]], agent_pos[[0, 2]])

        # Get upper and lower maximum view heights of agent at min_viewable_depth
        half_height = depth_val * np.tan(half_vfov_rad)
        upper_view_height = self.sensor_vert_pos + half_height
        lower_view_height = self.sensor_vert_pos - half_height

        if lower_view_height <= obj_pos[1] <= upper_view_height:
            return True
        return False

    def obj_bound_box_pts(
        self, obj_pos: np.ndarray, obj_dims: np.ndarray, agent_pos: np.ndarray
    ):
        r"""
        Generates points covering the dimensions of the object, with respect to the central object position
        """

        if obj_dims[2] < 0.1:
            # Centered on the object, not at the bottom.
            y_perturbs = [-0.5, 0.5]
        else:
            # Not Centered. Situated at the bottom of the object.
            y_perturbs = [0.5, 1]

        # Perturb along the xz plane: Five point (2 along each dimension, and one center point)
        xz_perturb_pts = np.array(
            [
                self.perturb_along_dim(obj_pos, dim, delta * obj_dims[dim])
                for dim in [0, 2]
                for delta in [-0.5, 0.5]
            ]
        )
        xz_perturb_pts = np.concatenate(
            (np.array(obj_pos)[np.newaxis, :], xz_perturb_pts)
        )

        # Perturb along the y axis and only keep the valid points within frame.
        y_perturb_pts = np.array(
            [self.perturb_along_dim(obj_pos, 1, i * obj_dims[1]) for i in y_perturbs]
        )
        y_perturb_pts = np.concatenate(
            (np.array(obj_pos)[np.newaxis, :], y_perturb_pts)
        )
        y_perturb_valid = [self.is_obj_in_frame(pt, agent_pos) for pt in y_perturb_pts]
        y_valid_pts = [
            y_perturb_pts[ind][1]
            for ind in range(len(y_perturb_pts))
            if y_perturb_valid[ind]
        ]

        # All points along the dimensions of the object
        obj_pts = [
            [xz_pt[0], y_pt, xz_pt[2]]
            for y_pt in y_valid_pts
            for xz_pt in xz_perturb_pts
        ]

        return obj_pts

    def get_obj_pixels(
        self, agent_pos: np.ndarray, obj_pos: np.ndarray, obj_dims: np.ndarray = None
    ):
        r"""
        Obtain the pixels corresponding to points along the object dimensions.
        """

        # Set agent height
        agent_pos[1] = self.sensor_vert_pos

        # Get Rotation Matrix
        quat_coeffs = self.face_object(obj_pos, agent_pos)
        quat = utils.quat_from_coeffs(quat_coeffs)
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        rot_mat = rot.as_matrix()

        if obj_dims is not None:
            obj_px = []
            obj_pt_depth = []

            # Gets positions of points around the object dimensions
            obj_dim_pts = self.obj_bound_box_pts(obj_pos, obj_dims, agent_pos)

            for pos in obj_dim_pts:

                rel_pos = pos - agent_pos  # Position of object wrt agent
                rel_pos = rel_pos @ rot_mat  # Rotates to face the object

                px = self.rel_pos_to_px(rel_pos)  # Pixel location from the position

                if (0 <= px[0] < self.sensor_h) and (0 <= px[1] < self.sensor_w):
                    obj_px.append(px)
                    obj_pt_depth.append(abs(rel_pos[2]))

            return np.array(obj_px), np.array(obj_pt_depth)

        # If obj_dims is None, only calculate for object position
        rel_pos = obj_pos - agent_pos
        rel_pos = rel_pos @ rot_mat

        obj_px = self.rel_pos_to_px(rel_pos)
        obj_pt_depth = abs(rel_pos[2])

        if (0 <= obj_px[0] < self.sensor_h) and (0 <= obj_px[1] < self.sensor_w):
            return np.array(obj_px)[np.newaxis, :], np.array([obj_pt_depth])

        # print(f"Error: Object pixels fall out of the sensor dimensions! Cannot obtain viewpoint!")
        return [], []

    def is_a_viewpoint(
        self,
        pt: np.ndarray,
        obj_pos: np.ndarray,
        obj_dims: np.ndarray = None,
        display: bool = None,
    ):
        r"""
        Check if the give point is a viewpoint of the object at obj_pos.

        Args:
            - pt: Possible viewpoint position.
            - obj_pos: Object position.

        Returns:
            - is_viewpoint (bool)
            - Viewpoint rotation
        """

        # Rotates the agent to face the object
        view_pt_rot = self.face_object(obj_pos, pt)

        # Agent height is enforced
        # pt[1] = 0
        # pt[1] = obj_pos[1]

        # Gets depth observation
        obs = self.obs_at_pose(pos=pt, rot=view_pt_rot)
        depth = obs["depth_sensor"]

        # Pixel locations corresponding to the positions around the object dimensions
        obj_dim_pixels, obj_dim_depth = self.get_obj_pixels(
            np.array(pt), np.array(obj_pos), obj_dims
        )
        # if len(obj_dim_pixels) == 0: return False, view_pt_rot

        # Depth heuristic: accept if a projected sample has no closer surface.
        for px, target_depth in zip(obj_dim_pixels, obj_dim_depth):

            observed_depth = depth[px[0], px[1]]

            if display:
                self.display_sample(
                    obs["color_sensor"],
                    obs["semantic_sensor"],
                    obs["depth_sensor"],
                    plot_pts=[px[::-1]],
                )

            if observed_depth < (target_depth):
                continue
            else:
                return True, view_pt_rot
        return False, view_pt_rot

    def view_pts_around(
        self,
        ref_pt: np.ndarray,
        view_pts_dist: float,
        max_radius: float,
        obj_pos: np.ndarray,
        obj_dims: np.ndarray = None,
        prev_ref_pt=None,
    ):
        r"""
        Generates View Points around a given (boundary) reference point (ref_pt).

        Args:
            - ref_pt: Boundary Point.
            - view_pts_dist: Distance between viewpoints.
            - max_radius: Maximum radius to generate viewpoints.
            - obj_pos: Position of object
            - prev_ref_pt: Previous Boundary Point used as reference to avoid redundant viewpoint generation.
        """

        view_pts = []
        step_radius = view_pts_dist
        valid_view_zones = []
        valid_degrees = []

        fail_non_nav = 0
        fail_prev_bound = 0
        fail_invalid = 0
        count_pts = 0

        # Iterates from the Maximum possible radius down to the minimum distance
        for radius in np.arange(max_radius, 0, -1 * step_radius):

            # Get angular step value
            perimeter = 2 * np.pi * radius
            num_concentric_pts = perimeter / view_pts_dist
            step_degree = math.ceil(360 / num_concentric_pts)

            # print(f'\nRadius: {radius}, Valid Zones: {valid_view_zones}, Num View Points: {len(view_pts)}, Step Degree: {step_degree}')

            for degree in np.arange(0, 360, step_degree):

                count_pts += 1

                # Potential Point
                rad = np.deg2rad(degree)
                dx, dz = radius * np.cos(rad), radius * np.sin(rad)
                radial_pt = [ref_pt[0] + dx, ref_pt[1], ref_pt[2] + dz]

                # Skip if point is close to the previous boundary point
                if prev_ref_pt is not None:
                    dist_to_prev_pt = self.dist_btw_pts(
                        np.array(radial_pt)[[0, 2]], prev_ref_pt[[0, 2]]
                    )
                    if dist_to_prev_pt <= (max_radius * 0.8):
                        fail_prev_bound += 1
                        continue

                if radius == max_radius:

                    # Check if this is a viewpoint
                    is_valid, view_pt_rot = self.is_a_viewpoint(
                        np.array(radial_pt), np.array(obj_pos), obj_dims
                    )

                    if is_valid:
                        valid_degrees.append(degree)

                # Skip if the point is not navigable
                if not self.sim.pathfinder.is_navigable(radial_pt):
                    fail_non_nav += 1
                    continue

                if radius < max_radius:

                    # If the point is inside valid zone, append and continue
                    skip_iter = False
                    for low_lim, high_lim in valid_view_zones:

                        if high_lim < low_lim:

                            if (low_lim <= degree <= 360) or (0 <= degree <= high_lim):
                                view_pts.append((radial_pt, view_pt_rot))
                                skip_iter = True
                                break

                        else:

                            if low_lim <= degree <= high_lim:
                                view_pts.append((radial_pt, view_pt_rot))
                                skip_iter = True
                                break

                    if skip_iter:
                        continue

                    # Check if point is a viewpoint
                    is_valid, view_pt_rot = self.is_a_viewpoint(
                        np.array(radial_pt), np.array(obj_pos), obj_dims
                    )

                    if not is_valid:
                        fail_invalid += 1
                        continue

                    # print(f" Radius: {radius}, Degree: {degree}")
                    view_pts.append((radial_pt, view_pt_rot))

            if (radius == max_radius) and (len(valid_degrees) > 1):

                valid_view_zones = self.group_angles(valid_degrees, step_degree)
                # print(f"Valid View Zones: {valid_view_zones}")

        # print(f"\n\n - Fail prev bound : {fail_prev_bound}\n - Fail non nav: {fail_non_nav}\n - Fail invalid: {fail_invalid}")
        # print(f"Total : {fail_prev_bound+fail_non_nav+fail_invalid} / {count_pts}")
        # print(f"Num viewpoints: {len(view_pts)}")

        if len(view_pts) > 1:

            range_of_angles = [
                self.face_object(view_pt[0], np.array(obj_pos), return_yaw=True)
                for view_pt in view_pts
            ]
            max_angle, min_angle = max(range_of_angles), min(range_of_angles)
            mean_angle = np.mean(range_of_angles)

            if min_angle <= mean_angle <= max_angle:
                clockwise = True
            else:
                clockwise = False

            # Check if this is a viewpoint
            _, ref_pt_rot = self.is_a_viewpoint(
                np.array(ref_pt), np.array(obj_pos), obj_dims
            )

            angle_obj_to_bound = self.face_object(ref_pt, obj_pos, return_yaw=True)

            if clockwise:
                if min_angle <= angle_obj_to_bound <= max_angle:
                    view_pts.append((ref_pt, ref_pt_rot))
            else:
                if (-180 < angle_obj_to_bound <= min_angle) or (
                    max_angle <= angle_obj_to_bound <= 180
                ):
                    view_pts.append((ref_pt, ref_pt_rot))

        return np.array(view_pts, dtype=object)

    def generate_view_pts(
        self,
        obj_pos: List,
        obj_dims: np.ndarray = None,
        boundary_pts_dist=0.5,
        boundary_check_radius=2,
        dist_btw_view_pts=0.2,
        view_pts_max_radius=1,
    ):

        # Historical sampling heuristic: narrow horizontal bounds before probing
        # visibility. These are probe dimensions, not the full object bounds.
        obj_dims = [obj_dims[0] * 0.5, obj_dims[1] * 1, obj_dims[2] * 0.5]

        # Get boundary points (evenly-spaced)
        success, spaced_boundary_pts = self.boundary_around_obj(
            obj_pos=obj_pos,
            obj_dims=obj_dims,
            pts_dist=boundary_pts_dist,
            keep_final_pt=True,
            _shoot_till=boundary_check_radius,
        )

        if not success:
            print(f"Could not obtain Boundary Points. Skipping...")
            return []

        # Generate Viewpoints around the Boundary
        ind = 0
        view_pts = []
        for bound_pt in spaced_boundary_pts:

            bound_view_pts = self.view_pts_around(
                ref_pt=bound_pt,
                view_pts_dist=dist_btw_view_pts,
                max_radius=view_pts_max_radius,
                obj_pos=obj_pos,
                obj_dims=obj_dims,
                prev_ref_pt=spaced_boundary_pts[ind - 1],
            )

            view_pts = self.concat_arr(view_pts, bound_view_pts)
            ind += 1

        return view_pts

    ## Useful Methods
    def _get_floor_height(self, search_center: np.ndarray) -> float:
        """Reference: OVON pose_sampler.py"""

        point = np.asarray(search_center)[:, None]
        snapped = self.sim.pathfinder.snap_point(point)

        # the centroid should not be lower than the floor
        tries = 0
        while point[1, 0] < snapped[1]:
            point[1, 0] -= 0.05
            snapped = self.sim.pathfinder.snap_point(point)
            tries += 1
            if tries > 40:  # trace 2.0m down.
                break

        return snapped[1]

    @staticmethod
    def dist_btw_pts(pt_0, pt_1):
        if not isinstance(pt_0, np.ndarray):
            pt_0 = np.array(pt_0)
        if not isinstance(pt_1, np.ndarray):
            pt_1 = np.array(pt_1)
        return np.linalg.norm(pt_1 - pt_0)

    @staticmethod
    def generate_pts_btw(pt_1, pt_2, num_pts):
        r"""
        Generate n-dim points between
        """

        assert len(pt_1) == len(pt_2), "Both points should be of same size!"
        assert num_pts >= 2, "Number of points requested is less than 2!"
        num_dims = len(pt_1)

        next_coord = lambda param, dim: (1 - param) * pt_1[dim] + (param) * pt_2[dim]
        param_vals = np.linspace(0, 1, num_pts)

        return [
            [next_coord(param, dim) for dim in range(num_dims)] for param in param_vals
        ]

    @staticmethod
    def concat_arr(arr_main: np.ndarray, arr_add: np.ndarray):
        r"""
        Concatenates two arrays
        """
        if len(arr_add) == 0:
            return arr_main
        if len(arr_main) == 0:
            return np.array(arr_add)
        return np.concatenate((np.array(arr_main), np.array(arr_add)))

    @staticmethod
    def quat_coeffs_to_yaw(quat_coeffs: List):
        r"""
        Converts Quatertion Coefficients to Yaw Angle.
        """

        quat = utils.quat_from_coeffs(quat_coeffs)

        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        rot_vec = rot.as_rotvec(degrees=True)
        return rot_vec[1]

    def get_vfov(self):
        r"""
        Obtains Vertical Field of Vision (VFOV) using HFOV, Sensor Resolution
        """

        hfov_rad = np.deg2rad(self.hfov)
        vfov_rad = 2 * math.atan(
            math.tan(hfov_rad / 2) * (self.sensor_h / self.sensor_w)
        )

        # return np.rad2deg(vfov_rad)
        return vfov_rad

    def rel_pos_to_px(self, rel_pos: List):
        r"""
        Projects 3D coordinate of object wrt agent position (rel_pos) onto the sensor frame.
        """

        delta_px_y = np.round(self.focus * rel_pos[1] / abs(rel_pos[2])).astype(int)
        delta_px_x = np.round(self.focus * rel_pos[0] / abs(rel_pos[2])).astype(int)

        px_y = (self.sensor_h // 2) - delta_px_y
        px_x = (self.sensor_w // 2) - delta_px_x

        return (px_y, px_x)

    ## Plot Methods
    @staticmethod
    def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown

    @staticmethod
    def display_sample(
        rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), plot_pts=None
    ):
        from habitat_sim.utils.common import d3_40_colors_rgb

        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        arr = [rgb_img]
        titles = ["rgb"]
        if semantic_obs.size != 0:
            semantic_img = Image.new(
                "P", (semantic_obs.shape[1], semantic_obs.shape[0])
            )
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")
            arr.append(semantic_img)
            titles.append("semantic")

        if depth_obs.size != 0:
            depth_img = Image.fromarray(
                (depth_obs / 10 * 255).astype(np.uint8), mode="L"
            )
            arr.append(depth_img)
            titles.append("depth")

        plt.figure(figsize=(12, 8))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i + 1)
            ax.axis("off")
            ax.set_title(titles[i])

            if plot_pts is not None:
                for pt in plot_pts:
                    ax.plot(pt[0], pt[1], "ro")

            plt.imshow(data)
        plt.show(block=False)

    @staticmethod
    def display_map(topdown_map, key_points=None, with_line=False):
        plt.figure(figsize=(12, 5))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")

        plt.imshow(topdown_map)

        # plot points on map
        start_col = "#e07a5f"
        end_col = "#81b29a"
        mid_col = "#f2cc8f"  # "yellow"
        if key_points is not None:
            for count, point in enumerate(key_points):
                if count == 0:
                    col = start_col
                elif count == len(key_points) - 1:
                    col = end_col
                else:
                    col = mid_col

                plt.plot(
                    point[0], point[1], marker="o", markersize=5, alpha=0.8, color=col
                )
                if count > 0 and with_line:
                    plt.plot(
                        [key_points[count][0], key_points[count - 1][0]],
                        [key_points[count][1], key_points[count - 1][1]],
                        linestyle="dashed",
                        linewidth=2,
                        color="#f4f1de",
                        alpha=0.5,
                    )

        plt.show(block=False)

    def plot_topdown_with_pts(
        self, pts, meters_per_pixel, with_line=False, snap_points=False
    ):

        snapped_pts = [self.sim.pathfinder.snap_point(pt) for pt in pts]

        xy_vis_points = self.convert_points_to_topdown(
            self.sim.pathfinder, snapped_pts if snap_points else pts, meters_per_pixel
        )

        top_down_map = maps.get_topdown_map(
            self.sim.pathfinder,
            height=snapped_pts[0][1],
            meters_per_pixel=meters_per_pixel,
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]

        self.display_map(top_down_map, key_points=xy_vis_points, with_line=with_line)

    def plot_agent_at(self, agent_pos, obj_pos=None, force_agent_height=True):

        if obj_pos is not None:
            rot = self.face_object(obj_pos, agent_pos)
        else:
            rot = None

        if force_agent_height:
            agent_pos[1] = 0

        obs = self.obs_at_pose(agent_pos, rot)
        self.display_sample(obs["color_sensor"], depth_obs=obs["depth_sensor"])


if __name__ == "__main__":

    view_pts_generator = ViewPoints_Generator()

    # Scene: 102344094, Obj Name: console table
    # obj_position = [-8.141680717468262, -5.266547375981645e-08, 0.883579976345299]
    # obj_dims = [1.2000000476837158, 0.75, 0.3499999940395355]

    # Scene: 106366410_174226806, Obj Name: bed, Britte California King Upholstered Headboard With Metal Bed Frame
    obj_position = [3.4664440155029297, -2.6885864201631193e-08, 0.4510699241715699]
    obj_dims = [2.268355369567871, 1.3969999551773074, 1.9125553965568542]

    view_pts = view_pts_generator.generate_view_pts(
        obj_pos=obj_position,
        obj_dims=obj_dims,
        # boundary_pts_dist = 0.2, boundary_check_radius = 2,
        # dist_btw_view_pts = 0.05, view_pts_max_radius = 1
    )
    print(
        f"\nGenerated {len(view_pts)} viewpoints for object at position {obj_position}."
    )
