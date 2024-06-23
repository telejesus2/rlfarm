import copy

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2 import SawyerButtonPressEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_basketball_v2 import SawyerBasketballEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_dial_turn_v2 import SawyerDialTurnEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_close_v2 import SawyerDrawerCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_peg_insertion_side_v2 import SawyerPegInsertionSideEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import SawyerReachEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_sweep_into_goal_v2 import SawyerSweepIntoGoalEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2

from rlfarm.envs.metaworld.metaworld_env import MetaWorldEnv
from rlfarm.envs.metaworld.recorder_metaworld_env import RecorderMetaWorldEnv


AVAILABLE_TASKS = { # task_name: task_class
    'button_press': SawyerButtonPressEnvV2,
    'basketball': SawyerBasketballEnvV2,
    'dial_turn': SawyerDialTurnEnvV2,
    'drawer_close': SawyerDrawerCloseEnvV2,
    'peg_insertion_side': SawyerPegInsertionSideEnvV2,
    'pick_place': SawyerPickPlaceEnvV2,
    'push': SawyerPushEnvV2,
    'reach': SawyerReachEnvV2,
    'sweep_into_goal': SawyerSweepIntoGoalEnvV2,
    'window_open': SawyerWindowOpenEnvV2,
}


def make_env(config):   
    env_config = config['environment']
    task_name = env_config['task']['class']
    task_kwargs = env_config['task'].get('kwargs', {})
    assert task_name in AVAILABLE_TASKS
    

    class_ = MetaWorldEnv if not env_config['record'] else RecorderMetaWorldEnv
    return class_(
        task_name = task_name,
        task_class = AVAILABLE_TASKS[task_name],
        **env_config['kwargs'],
    )