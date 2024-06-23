from typing import Dict, Tuple, List

import numpy as np
import torch

from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition, \
    EndEffectorPoseViaPlanning, FlatEndEffectorPoseViaPlanning, \
    EndEffectorPoseViaIK, FlatEndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete, Closed, DiscreteHold
from rlbench.action_modes.action_mode import MoveArmThenGripper, Primitives, XYPrimitives
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.demo import Demo

from rlfarm.envs.env import ActionSpace
from rlfarm.utils.logger import Summary, ScalarSummary


### MoveArmThenGripper ########################################################################

ARM_ACTION_MODES = [ # (action, action_shape)
    (JointVelocity(), lambda x: SUPPORTED_ROBOTS[x][2]),
    (JointPosition(absolute_mode=True), lambda x: SUPPORTED_ROBOTS[x][2]),
    (EndEffectorPoseViaPlanning(
        absolute_mode=True, frame='world', collision_checking=False),
        lambda x: 7),
    (EndEffectorPoseViaIK(
        absolute_mode=True, frame='world', collision_checking=False),
        lambda x: 7),
    (FlatEndEffectorPoseViaPlanning(
        absolute_mode=True, frame='world', collision_checking=False, linear_only=True),
        lambda x: 2),
    (FlatEndEffectorPoseViaIK(
        absolute_mode=True, frame='world', collision_checking=False),
        lambda x: 2),         
]
GRIPPER_ACTION_MODES = [
    (Discrete(), lambda x: 1),
    (Closed(), lambda x: 0),
    (DiscreteHold(), lambda x: 1),
]

class MoveArmThenGripperActionSpace(ActionSpace):
    def __init__(self, arm_action_mode, gripper_action_mode, arm_action_size, gripper_action_size):
        self._arm_action_mode = arm_action_mode
        self._gripper_action_mode = gripper_action_mode
        self.arm_action_size = arm_action_size
        self.gripper_action_size = gripper_action_size
        self.action_shape = (np.prod(arm_action_size) + np.prod(gripper_action_size),)
        self.action_size = self.action_shape[0]

        # we assume gripper action is at the end
        if self._gripper_action_mode in [0,2]:
            self._min_gripper_action = 0
            self._max_gripper_action = 1
        elif self._gripper_action_mode == 1:
            self._min_gripper_action = self._max_gripper_action = None

        if self._arm_action_mode == 0:
            # maybe should be smaller for three first non-wrist joints
            self._min_arm_action = self.arm_action_size * [-1]
            self._max_arm_action = self.arm_action_size * [1]
        elif self._arm_action_mode == 1:
            # TODO should define joint limits here
            self._min_arm_action = self.arm_action_size * [-1]
            self._max_arm_action = self.arm_action_size * [1]
        elif self._arm_action_mode == 2:
            self._min_arm_action = self.arm_action_size * [0]
            self._max_arm_action = self.arm_action_size * [0]
            # set workspace limits
            self._min_arm_action[:3] = np.array([-0.07634869, -0.44309086, 0.7579899])
            self._max_arm_action[:3] = np.array([0.53690785, 0.45407084, 1.2445989])
            # set quaternion min max
            self._min_arm_action[3:7] = np.array([-1, -1, -1, 0])
            self._max_arm_action[3:7] = np.array([1, 1, 1, 1])
        elif self._arm_action_mode == 4:
            # set workspace limits
            self._min_arm_action = [0.05, -0.4]
            self._max_arm_action = [0.45, 0.4]
            # self._min_arm_action = [0.1,-0.2,0.76]
            # self._max_arm_action = [0.35,0.2,0.96]
        elif self._arm_action_mode == 5:
            # set workspace limits
            self._min_arm_action = [-0.1, -0.1]
            self._max_arm_action = [0.1, 0.1]
        else:
            raise NotImplementedError()

    def normalize(self, action: torch.tensor) -> torch.tensor:
        if self._arm_action_mode == 2:
            action = torch.cat([
                action[:, :3], self._normalize(action[:, 3:7]), action[:, 7:]
            ], dim=-1)
        return action

    def _normalize(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def sample(self) -> torch.tensor:
        action = [torch.FloatTensor(1).uniform_(min_action, max_action) 
            for (min_action, max_action) in zip(self._min_arm_action, self._max_arm_action)]
        if self.gripper_action_size > 0:
            action += [torch.FloatTensor(1).uniform_(self._min_gripper_action, self._max_gripper_action)]
        action = torch.cat(action)

        return self.normalize(action.view(1, -1))

    def get_action_min_max(self, action_min_max: Tuple[np.ndarray] = None) -> Tuple[np.ndarray]:
        # make bounds a little bigger
        if action_min_max is None:
            if self.gripper_action_size == 1:
                action_min_max = np.array(self._min_arm_action + [self._min_gripper_action]).astype(np.float32), \
                                 np.array(self._max_arm_action + [self._max_gripper_action]).astype(np.float32)
            elif self.gripper_action_size == 0:
                 action_min_max = np.array(self._min_arm_action).astype(np.float32), \
                                  np.array(self._max_arm_action).astype(np.float32)
            else:
                pass # TODO

        else:
            # gripper
            if self.gripper_action_size > 0: 
                action_min_max[0][-self.gripper_action_size:] = self._min_gripper_action
                action_min_max[1][-self.gripper_action_size:] = self._max_gripper_action
            # arm
            if self._arm_action_mode == 0:
                action_min_max[0][:self.arm_action_size] -= np.fabs(action_min_max[0][:self.arm_action_size]) * 0.2
                action_min_max[1][:self.arm_action_size] += np.fabs(action_min_max[1][:self.arm_action_size]) * 0.2
            elif self._arm_action_mode == 1:
                # TODO what would make sense here?
                action_min_max[0][:self.arm_action_size] -= np.fabs(action_min_max[0][:self.arm_action_size]) * 0.2
                action_min_max[1][:self.arm_action_size] += np.fabs(action_min_max[1][:self.arm_action_size]) * 0.2
            elif self._arm_action_mode == 2:
                # action_min_max[0][0:3] -= np.fabs(action_min_max[0][0:3]) * 0.2
                # action_min_max[1][0:3] += np.fabs(action_min_max[1][0:3]) * 0.2
                action_min_max[0][3:7] = np.array([-1, -1, -1, 0])
                action_min_max[1][3:7] = np.array([1, 1, 1, 1])
        return action_min_max

    def extract_actions_from_demo(self, demo: Demo):
        # TODO should account for gripper_action_mode as well
        if self._arm_action_mode == 0:
            acs = [np.concatenate([ob.joint_velocities, [ob.gripper_open]], axis=-1)
                   for ob in demo[1:]]
        elif self._arm_action_mode == 1:
            acs = [np.concatenate([ob.joint_positions, [ob.gripper_open]], axis=-1)
                   for ob in demo[1:]]
        elif self._arm_action_mode in [2,4]:
            # acs = [np.concatenate([ob.gripper_pose, [ob.gripper_open]], axis=-1)
            #        for ob in demo[1:]]
            # TODO should replace this
            acs = [ob.misc['action'] for ob in demo[1:]]
        else:
            raise NotImplementedError()

        return acs

    def log_actions(self, actions: torch.tensor, prefix: str) -> List[Summary]:
        return []


### Primitives ##############################################################################

class PrimitivesActionSpace(ActionSpace):
    def __init__(self):
        self.action_size = 6
        self._num_primitives = 3

        # TODO these are hard-coded values for the pick_and_lift_no_distractors task
        self._min_arm_action = [0, 0, 0, 0.07599998, -0.28950004, 0]
        self._max_arm_action = [1, 1, 1, 0.37599999, 0.27549996, np.pi/2]

    def normalize(self, action: torch.tensor) -> torch.tensor:
        return action

    def _normalize(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def sample(self) -> torch.tensor:
        action = [torch.FloatTensor(1).uniform_(min_action, max_action) 
            for (min_action, max_action) in zip(self._min_arm_action, self._max_arm_action)]
        action = torch.cat(action)

        return self.normalize(action.view(1, -1))

    def get_action_min_max(self, action_min_max: Tuple[np.ndarray] = None) -> Tuple[np.ndarray]:
        action_min_max = np.array(self._min_arm_action).astype(np.float32), \
                        np.array(self._max_arm_action).astype(np.float32)
        return action_min_max

    def extract_actions_from_demo(self, demo: Demo):
        acs = [ob.misc['action'] for ob in demo[1:]]
        return acs

    def log_actions(self, actions: torch.tensor, prefix: str) -> List[Summary]:
        maximum = torch.max(actions[:,:self._num_primitives], 1)[0].mean()
        minimum = torch.min(actions[:,:self._num_primitives], 1)[0].mean()
        return [
            ScalarSummary(prefix + '/actions_primitive_max', maximum),
            ScalarSummary(prefix + '/actions_primitive_min', minimum),
            ScalarSummary(prefix + '/actions_primitive_delta', maximum - minimum),            
        ]

class XYPrimitivesActionSpace(PrimitivesActionSpace):
    def __init__(self):
        self.action_size = 5
        self._num_primitives = 2

        # TODO these are hard-coded values for the pick_and_lift_no_distractors task
        self._min_arm_action = [0, 0, 0.07599998, -0.28950004, 0]
        self._max_arm_action = [1, 1, 0.37599999, 0.27549996, np.pi/2]

### Builder ##############################################################################

def make_action_space(config: dict):
    action_config = config['environment']['action']
    action_class = action_config['class']
    robot_config = config['environment']['kwargs']['robot_config']
    action_space, action_mode = None, None

    if action_class == 'move-arm-then-gripper':
        action_config = action_config['kwargs']

        action_mode = MoveArmThenGripper(
            ARM_ACTION_MODES[action_config['arm_action']][0],
            GRIPPER_ACTION_MODES[action_config['gripper_action']][0])

        arm_action_shape = ARM_ACTION_MODES[action_config['arm_action']][1](robot_config)
        gripper_action_shape = GRIPPER_ACTION_MODES[action_config['gripper_action']][1](robot_config)
        action_space = MoveArmThenGripperActionSpace(
            action_config['arm_action'], action_config['gripper_action'],
            arm_action_shape, gripper_action_shape)

    elif action_class == 'primitives':
        action_mode = Primitives(**action_config['kwargs'])
        action_space = PrimitivesActionSpace()

    elif action_class == 'xyprimitives':
        action_mode = XYPrimitives(**action_config['kwargs'])
        action_space = XYPrimitivesActionSpace()

    return action_space, action_mode