import logging
import random
from typing import Dict, Tuple, List

import numpy as np
import torch

from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.task import Task
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench import utils
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.tasks import PushButton, PressSwitch, ReachTarget, ReachTargetNoDistractors, ReachButton, \
                          SlideBlockToTarget, StackBlocks, PickAndLiftNoDistractors, PickAndPlace

from rlfarm.envs.env import Env, ActionSpace
from rlfarm.utils.transition import Transition
from rlfarm.utils.logger import Summary, ImageSummary

# AVAILABLE_OBSERVATIONS = ['low_dim_state', 'rgb_state']
AVAILABLE_CAMERAS = ['front', 'wrist']
AVAILABLE_TASKS = { # task_name: (task_class, task_low_dim_state_len)
    'reach_target': (ReachTarget, 3),
    'reach_target_no_distractors': (ReachTargetNoDistractors, 3),
    'pick_and_lift_no_distractors': (PickAndLiftNoDistractors, 5),
    'slide_block_to_target': (SlideBlockToTarget, 6),
    'push_button': (PushButton, 3),
    'reach_button': (ReachButton, 3),
    'press_switch': (PressSwitch, 3),
    'pick_and_place': (PickAndPlace, 5)
}


class RLBenchEnv(Env):
    def __init__(self,
                 task_name: str,
                 task_class: Task,
                 task_kwargs: dict,
                 action_mode: ActionMode,
                 action_space: ActionSpace,
                 observation_config: ObservationConfig,
                 demo_observation_config: ObservationConfig,
                 robot_config: str,
                 dataset_root: str = '',
                 headless=True,
                 max_episode_steps=np.inf,
                 variations: List[int] = [0],
                 swap_variation_every: int = 1,
                 stack_vision_channels=True,
                 channels_first=False,
                 reward_scale=100.0,
                 reset_to_demo_ratio: float = 0.0,
                 state_includes_variation_index: bool = False,
                 state_includes_remaining_time: bool = False,
                 state_includes_previous_action: bool = False,
    ):
        super(RLBenchEnv, self).__init__()
        self._stack_vision_channels = stack_vision_channels
        self._channels_first = channels_first
        self._task_name = task_name
        self._task_class = task_class
        self._variation_list = variations
        self._variation = variations[0]
        self._swap_variation_every = swap_variation_every
        self._reward_scale = reward_scale
        self._reset_to_demo_ratio = reset_to_demo_ratio
        self._state_includes_remaining_time = state_includes_remaining_time
        self._state_includes_previous_action = state_includes_previous_action
        self._state_includes_variation_index = state_includes_variation_index
        self._obs_config = observation_config
        self._demo_obs_config = demo_observation_config
        self._task_kwargs = task_kwargs

        # utilities
        self._i = 0
        self._prev_action = None
        self._episodes_this_variation = 0

        # define environment
        # visual_random = VisualRandomizationConfig(
        #     image_directory='./experiment_textures/train/top10',
        #         whitelist = ['Floor', 'Roof', 'Wall1', 'Wall2', 'Wall3', 'Wall4', 'diningTable_visible'],
        #         apply_arm = False,
        #         apply_gripper = False,
        #         apply_floor = True)
        self._env = Environment(action_mode, 
                                obs_config=observation_config,
                                robot_setup=robot_config, 
                                dataset_root=dataset_root,
                                headless=headless,
    #                             randomize_every=RandomizeEvery.EPISODE,
    #                             visual_randomization_config=VisualRandomizationConfig(
    # image_directory='/home/caor/Documents/MyRLBench/tests/unit/assets/textures')
        )
        
        # properties
        self.state_shape, self.state_dtype = self._extract_ob_specs(observation_config)
        self.action_space = action_space
        self.action_shape = (action_space.action_size,)
        self.action_dtype = np.float32
        self.max_episode_steps = max_episode_steps
        self.num_demos_per_variation = self._env.get_num_demos(self._task_name,
            variation_number=variations[0]) # assumes all variations have same amount of demos
        self.average_demo_length = None # should be overriden (see buffers.replay.builder.make_buffer())
        logging.info("Number of demos per variation available: %d" % self.num_demos_per_variation)
        logging.info("State shape: %s" % str(self.state_shape))
        
    def launch(self, agent=None):
        self._env.launch()
        self._task = self._env.get_task(self._task_class)
        self._task._scene.init_task(self._task_kwargs)
        self._task.set_variation(self._variation)

    @property
    def unwrapped(self):
        return self

    @property
    def history_len(self) -> int:
        return 1

    @property
    def num_variations(self) -> int:
        return len(self._variation_list)

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def active_task_id(self) -> int:
        # returns task index, not the actual id
        return self._variation_list.index(self._variation)

    def render(self, mode='human', **kwargs):
        pass

    def seed(self, seed):
        # TODO
        pass

    def close(self):
        self._task = None
        return self._env.shutdown()

    def step(self, action: np.ndarray) -> Transition:
        obs = self._previous_obs_dict  # in case action fails.
        try:
            obs, reward, terminate, info = self._task.step(action)
            obs = self._extract_ob(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            reward, terminate, info = 0, True, {"error": True}
        self._i += 1
        self._prev_action = action
        return Transition(obs, reward * self._reward_scale, terminate, info)

    def reset(self):
        logging.info("Episode reset...")
        # sample variation
        self._episodes_this_variation += 1
        if self._episodes_this_variation == self._swap_variation_every:
            self._set_new_variation()
            self._episodes_this_variation = 0
        # reset task
        if self._reset_to_demo_ratio > 0 and random.random() < self._reset_to_demo_ratio:
            descriptions, obs =  self._reset_to_demo()
        else:
            descriptions, obs = self._task.reset()
        del descriptions  # Not used.
        self._i = 0
        self._prev_action = None
        # reset summaries
        self.episode_summaries = [] # see runners.samplers.rollout_generator.RolloutGenerator

        # track previous obs (in case action fails in step())
        self._previous_obs_dict = self._extract_ob(obs)

        return self._previous_obs_dict

    def _set_new_variation(self):
        if len(self._variation_list) > 1:
            self._variation = np.random.choice(self._variation_list)
            self._task.set_variation(self._variation)

    def _extract_ob(self, ob: Observation, variation=None, t=None, prev_action=None, 
                    include_low_dim=True, include_rgb=True, include_point_cloud=True) -> Dict[str, np.ndarray]:  
        
        # TODO this is for arm
        # turn gripper quaternion to be positive w
        if ob.gripper_pose is not None and ob.gripper_pose[-1] < 0:
            ob.gripper_pose[3:] *= -1.0
        # clip joint positions
        if ob.gripper_joint_positions is not None:
            ob.gripper_joint_positions = np.clip(
                ob.gripper_joint_positions, 0., 0.04)
        
        ob_dict = vars(ob)
        new_ob = {}
        if variation is None:
            variation = self._variation

        # add low dim state
        if include_low_dim:
            key = 'low_dim_state'
            low_dim_state = np.array(ob.get_low_dim_data(), dtype=np.float32)
            if self._state_includes_variation_index:
                low_dim_state = np.concatenate([low_dim_state, [variation]]).astype(np.float32)
            if self._state_includes_remaining_time:
                tt = 1. - ((self._i if t is None else t) / self.max_episode_steps)
                low_dim_state = np.concatenate([low_dim_state, [tt]]).astype(np.float32)
            if self._state_includes_previous_action:
                pa = self._prev_action if prev_action is None else prev_action
                pa = np.zeros(self.action_shape) if pa is None else pa
                low_dim_state = np.concatenate([low_dim_state, pa]).astype(np.float32)
            new_ob[key] = low_dim_state 

        # add rgb image state
        if include_rgb:
            rgb_data = []
            for prefix in AVAILABLE_CAMERAS:
                key = prefix + "_rgb"
                if ob_dict[key] is not None:
                    if not self._stack_vision_channels:
                        new_ob[key] = ob_dict[key].astype(np.uint8)
                    else:
                        rgb_data.append(ob_dict[key])
            if len(rgb_data) > 0 and self._stack_vision_channels:
                    key = "rgb_state"
                    new_ob[key] = np.concatenate(rgb_data, axis=-1).astype(np.uint8)

        # add point cloud state
        if include_point_cloud:
            for prefix in AVAILABLE_CAMERAS:
                key = prefix + "_point_cloud" # e.g. front_point_cloud
                if ob_dict[key] is not None:
                    new_ob[key] = ob_dict[key].astype(np.float32)
                    key = prefix + "_camera_extrinsics"
                    new_ob[key] = ob.misc[key]
                    key = prefix + "_camera_intrinsics"
                    new_ob[key] = ob.misc[key]

        if self._channels_first:
            new_ob = {k: np.transpose(
                v, [2, 0, 1]) if v.ndim == 3 else v
                        for k, v in new_ob.items()}

        return new_ob

    def _extract_ob_specs(self, ob_config: ObservationConfig):
        ob_config_dict = vars(ob_config)
        ob_shape = {}
        ob_dtype = {}

        # add low dim state
        key = 'low_dim_state'
        low_dim_state_len = 0
        if ob_config.joint_velocities:           low_dim_state_len += 6
        if ob_config.joint_positions:            low_dim_state_len += 6
        if ob_config.joint_forces:               low_dim_state_len += 6
        if ob_config.gripper_open:               low_dim_state_len += 1
        if ob_config.gripper_pose:               low_dim_state_len += 7
        if ob_config.gripper_joint_positions:    low_dim_state_len += 2
        if ob_config.gripper_touch_forces:       low_dim_state_len += 2
        if ob_config.task_low_dim_state:         low_dim_state_len += AVAILABLE_TASKS[self._task_name][1]
        if self._state_includes_variation_index: low_dim_state_len += 1 
        if self._state_includes_remaining_time:  low_dim_state_len += 1
        if self._state_includes_previous_action: low_dim_state_len += self.action_shape
        ob_shape[key], ob_dtype[key] = (low_dim_state_len,), np.float32

        def im_shape(im_size, channels=3):
            if self._channels_first:
                return (channels,) + tuple(im_size)
            else:
                return tuple(im_size) + (channels,)

        # add rgb image state
        num_cameras = 0
        image_size = None
        for prefix in AVAILABLE_CAMERAS:
            camera_config = ob_config_dict[prefix + "_camera"]
            if camera_config is not None and camera_config.rgb:
                num_cameras += 1
                image_size = camera_config.image_size # assumes all cameras have same image size
                if not self._stack_vision_channels:
                    key = prefix + "_rgb" # e.g. front_rgb
                    ob_shape[key], ob_dtype[key] = im_shape(camera_config.image_size), np.uint8
        if self._stack_vision_channels and num_cameras > 0:
            key = "rgb_state"
            ob_shape[key], ob_dtype[key] = im_shape(image_size, 3 * num_cameras), np.uint8

        # add point cloud state
        for prefix in AVAILABLE_CAMERAS:
            camera_config = ob_config_dict[prefix + "_camera"]
            if camera_config is not None and camera_config.point_cloud:
                key = prefix + "_point_cloud" # e.g. front_point_cloud
                ob_shape[key], ob_dtype[key] = im_shape(camera_config.image_size), np.float32
                key = prefix + "_camera_extrinsics"
                ob_shape[key], ob_dtype[key] = (4, 4), np.float32
                key = prefix + "_camera_intrinsics"
                ob_shape[key], ob_dtype[key] = (3, 3), np.float32

        return ob_shape, ob_dtype

    #============================================================================================#
    # DEMOS
    #============================================================================================#

    def _reset_to_demo(self):
        random_seed = np.random.get_state()
        d = self.get_demos(variation=self._variation, random_selection=True)[0]
        ob = np.random.choice(d[:-1], 1)[0]
        d.restore_state()
        out = self._task.reset(ob.joint_positions)
        np.random.set_state(random_seed)
        return out

    def get_demos(self, index=None, random_selection=False, amount=1, variation=None):
        demos = []
        assert random_selection or index is not None
        if variation is None and len(self._variation_list) == 1:
            variation = self._variation

        if variation is None: # get all variations
            for i in self._variation_list:
                demos += utils.get_stored_demos(
                    amount, False, self._env._dataset_root, i, self._task_name,
                    self._demo_obs_config, random_selection=random_selection, from_episode_number=index)
        else:                 # get one variation
            demos += utils.get_stored_demos(
                amount, False, self._env._dataset_root, variation, self._task_name,
                self._demo_obs_config, random_selection=random_selection, from_episode_number=index)

        return demos

    def extract_demo(self, demo: Demo):
        "usually called on demos returned by self.get_demos()"

        # extract the actions
        acs = self.action_space.extract_actions_from_demo(demo)

        # extract the rewards
        rews = self._task_class.reward_from_demo(demo, **self._task_kwargs['reward_kwargs'])
        rews = [r * self._reward_scale for r in rews]

        # remove entries if necessary
        for ob in demo:                
            if not self._obs_config.joint_velocities:           ob.joint_velocities = None
            if not self._obs_config.joint_positions:            ob.joint_positions = None
            if not self._obs_config.joint_forces:               ob.joint_forces = None
            if not self._obs_config.gripper_open:               ob.gripper_open = None
            if not self._obs_config.gripper_pose:               ob.gripper_pose = None
            if not self._obs_config.gripper_joint_positions:    ob.gripper_joint_positions = None
            if not self._obs_config.gripper_touch_forces:       ob.gripper_touch_forces = None
            if not self._obs_config.task_low_dim_state:         ob.task_low_dim_state = None 

        # extract the states
        variation = getattr(demo, 'variation_index', None)
        demo_emb, sums =  None, []
        obs = [self._extract_ob(ob, variation=variation, t=k,
                                prev_action=acs[k-1] if k>0 else None)
               for k, ob in enumerate(demo)]

        return obs, acs, rews, sums