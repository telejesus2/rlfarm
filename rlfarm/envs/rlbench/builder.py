import copy

from rlbench.observation_config import ObservationConfig, CameraConfig

from rlfarm.envs.rlbench.rlbench_env import RLBenchEnv, AVAILABLE_TASKS
from rlfarm.envs.rlbench.recorder_rlbench_env import RecorderRLBenchEnv
from rlfarm.envs.rlbench.rlbench_action_space import make_action_space


def make_observation_config(image_size: tuple,
                            front_rgb: bool,
                            wrist_rgb: bool,
                            front_point_cloud: bool,
                            joint_positions: bool,
                            joint_velocities: bool,
                            gripper_pose: bool,
                            gripper_open: bool,
                            low_dim: bool) -> ObservationConfig:
    cam_config = CameraConfig(image_size=image_size)
    obs_config = ObservationConfig(cam_config, 
        copy.deepcopy(cam_config), copy.deepcopy(cam_config),
        copy.deepcopy(cam_config), copy.deepcopy(cam_config))
    obs_config.joint_velocities = joint_velocities
    obs_config.joint_positions = joint_positions
    obs_config.gripper_open = gripper_open
    obs_config.gripper_pose = gripper_pose
    obs_config.front_camera.rgb = front_rgb
    obs_config.wrist_camera.rgb = wrist_rgb
    obs_config.front_camera.point_cloud = front_point_cloud
    if not front_rgb: obs_config.front_camera.depth = front_point_cloud # RLBench requirement: one of depth/rgb has to be true to access the point cloud
    obs_config.task_low_dim_state = low_dim
    return obs_config


def make_env(config, record=False):   
    env_config = config['environment']
    task_name = env_config['task']['class']
    task_kwargs = env_config['task'].get('kwargs', {})
    assert task_name in AVAILABLE_TASKS

    obs_config = make_observation_config(**env_config['observation'])
    demo_obs_config = make_observation_config(
        obs_config.front_camera.image_size,
        obs_config.front_camera.rgb, obs_config.wrist_camera.rgb,
        obs_config.front_camera.point_cloud,
        True, True, True, True, True)

    action_space, action_mode = make_action_space(config)
    
    class_ = RLBenchEnv if not record else RecorderRLBenchEnv
    return class_(
        task_name = task_name,
        task_class=AVAILABLE_TASKS[task_name][0],
        task_kwargs= task_kwargs,
        action_mode=action_mode,
        action_space=action_space,
        observation_config=obs_config,
        demo_observation_config=demo_obs_config,
        **env_config['kwargs'],
    )