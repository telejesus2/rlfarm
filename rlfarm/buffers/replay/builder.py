import logging
from functools import partial

import numpy as np

from rlfarm.envs.env import Env
from rlfarm.buffers.replay.replay_buffer import ReplayElement, ReplayBuffer
from rlfarm.buffers.replay.uniform_replay_buffer import UniformReplayBuffer
from rlfarm.buffers.replay.prioritized_replay_buffer import PrioritizedReplayBuffer
from rlfarm.buffers.replay.utils import store_transition, pack_episode
from rlfarm.buffers.replay.const import DEMO
from rlfarm.buffers.replay.callback import make_callback


def make_buffer(
    save_dir: str, env: Env, config: dict, callback_device
):
    main_config = config['buffer']['main_buffer']
    demo_config = config['buffer']['demo_buffer']
    use_demo_buffer = demo_config['batch_size_ratio'] > 0
    demo_info = main_config['use_demos'] or use_demo_buffer
    extra_replay_elements = []
    if demo_info:
        extra_replay_elements += [ReplayElement(DEMO, (), np.bool)]
    all_actions = []

    # get episode callback
    will_add_demos = use_demo_buffer or \
                     (main_config['use_demos'] and main_config['demos']['num_init_per_variation'])
    callback, callback_agent, state_shape, state_dtype = make_callback(
        config, callback_device, env.state_shape, env.state_dtype, should_init_agent=will_add_demos)

    # get info from algorithm config
    alg_config = config['agent']
    main_config['kwargs']['gamma'] = alg_config['kwargs']['gamma']

    # demo buffer
    demo_replay = None
    if use_demo_buffer:
        # get info from main buffer config
        for key in ['n_steps', 'gamma', 'history_len', 'max_sample_attempts']: 
            demo_config['kwargs'][key] = main_config['kwargs'][key]

        # set new batch sizes
        full_batch_size = main_config['kwargs']['batch_size']
        demo_config['kwargs']['batch_size'] = int(
            demo_config['batch_size_ratio'] * full_batch_size)
        main_config['kwargs']['batch_size'] = \
            full_batch_size - demo_config['kwargs']['batch_size']

        demo_replay, demo_actions = _make_buffer(
            save_dir + '_demos', env, state_shape, state_dtype, callback, callback_agent,
            demo_config, alg_config, 
            extra_replay_elements=extra_replay_elements, fill_with_demos=True)
        all_actions.extend(demo_actions)

        logging.info('Demo buffer size: %s. Total capacity: %s' % (
            demo_replay.add_count, demo_replay._replay_capacity))
        if demo_replay.is_full():
            raise RuntimeError('Demo buffer full, \
                increase replay_capacity so that no demos are wasted.')

    # main buffer
    explore_replay, explore_actions = _make_buffer(
        save_dir, env, state_shape, state_dtype, callback, callback_agent,
        main_config, alg_config, 
        extra_replay_elements=extra_replay_elements, fill_with_demos=main_config['use_demos'])
    all_actions.extend(explore_actions)

    # action statistics
    action_min_max = None
    if len(all_actions) > 0:
        action_min_max = np.min(all_actions, axis=0).astype(np.float32), \
                         np.max(all_actions, axis=0).astype(np.float32)        
    action_min_max = env.action_space.get_action_min_max(action_min_max)

    # demo statistics
    if len(all_actions) > 0:
        env.average_demo_length = len(all_actions) // ((
            main_config['demos']['num_init_per_variation'] * int(main_config['use_demos']) + 
            demo_config['demos']['num_init_per_variation'] * int(use_demo_buffer)
            ) * env.num_variations)
        logging.info("Average demo length: %d" % env.average_demo_length)

    del callback_agent
    return explore_replay, demo_replay, action_min_max, demo_info, callback
    

def _make_buffer(save_dir, env, state_shape, state_dtype, callback, callback_agent,
    buffer_config, alg_config, extra_replay_elements=[], fill_with_demos=False
):
    replay_class = UniformReplayBuffer
    if buffer_config['prioritized']:
        replay_class = PrioritizedReplayBuffer

    # only for algorithm ARM
    if alg_config['class'] == "arm":
        cameras = [alg_config["kwargs"]["camera"]]
        for cname in cameras:
            sname = '%s_pixel_coord' % cname
            assert sname not in state_shape
            state_shape[sname] = (2,)
            state_dtype[sname] = np.int32

    buffer = replay_class(
        save_dir=save_dir if buffer_config['use_disk'] else None,
        extra_replay_elements=extra_replay_elements,
        state_shape=state_shape,
        state_dtype=state_dtype,
        action_shape=env.action_shape,
        action_dtype=env.action_dtype,
        reward_shape=(),
        reward_dtype=np.float32,
        **buffer_config['kwargs'],
    )

    if alg_config['class'] == "arm":
        kwargs = {"cameras": cameras}
        fill_replay_fun = partial(_fill_replay_with_keypoint_demos, **kwargs) 
    else:
        fill_replay_fun = _fill_replay_with_demos
    actions =fill_replay_fun(
        buffer, env, callback, callback_agent, buffer_config) if fill_with_demos else []
    return buffer, actions


def _fill_replay_with_demos(replay: ReplayBuffer, env: Env, callback, callback_agent, buffer_config):
    # if buffer_config['demos']['augmentation']: # TODO demo augmentation
    #     raise NotImplementedError()
    logging.info('Filling replay with demos...')

    all_actions = []
    num_demos_per_variation = buffer_config['demos']['num_init_per_variation']
    assert num_demos_per_variation <= env.num_demos_per_variation, 'Requested too many initial demos.'
    demos = env.get_demos(index=0, amount=num_demos_per_variation)
    for demo in demos:
        all_actions.extend(_add_demo_to_replay(replay, env, demo, callback, callback_agent))

    logging.info('Replay filled with demos.')
    return all_actions


def _add_demo_to_replay(replay, env, demo, callback, callback_agent):
    obs, acs, rews, sums = env.extract_demo(demo)
    transitions = pack_episode(obs, acs, rews, {DEMO: True}, sums)
    if callback is not None:
        callback(transitions, env, callback_agent, [], 0, True)
    for transition in transitions: store_transition(transition, replay)
    return acs


###########################
# KEYPOINTS (USED FOR ARM)
###########################

import logging
from typing import List

from rlbench.demo import Demo
from rlbench.backend.observation import Observation


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                       last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    logging.debug('Found %d keypoints.' % len(episode_keypoints),
                  episode_keypoints)
    return episode_keypoints


def _point_to_pixel_index(
        point: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def _get_action(obs_tp1: Observation):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _add_keypoints_to_replay(
        cameras: List,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        env: Env,
        episode_keypoints: List[int]):
    prev_action = None
    obs = inital_obs
    all_actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(obs_tp1)
        all_actions.append(action)
        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * env._reward_scale if terminal else 0 # TODO
        obs_dict = env._extract_ob(obs, t=k, prev_action=prev_action)
        prev_action = np.copy(action)
        others = {DEMO: True}
        final_obs = {}
        for name in cameras:
            px, py = _point_to_pixel_index(
                obs_tp1.gripper_pose[:3],
                obs_tp1.misc['%s_camera_extrinsics' % name],
                obs_tp1.misc['%s_camera_intrinsics' % name])
            final_obs['%s_pixel_coord' % name] = [py, px]
        others.update(final_obs)
        others.update(obs_dict)
        timeout = False
        replay.add(action, reward, terminal, timeout, others)
        obs = obs_tp1  # Set the next obs
    # Final step
    obs_dict_tp1 = env._extract_ob(
        obs_tp1, t=k + 1, prev_action=prev_action)
    obs_dict_tp1.update(final_obs)
    replay.add_final(obs_dict_tp1)
    return all_actions


def _fill_replay_with_keypoint_demos(
        replay: ReplayBuffer, env: Env, callback, callback_agent, buffer_config, cameras=[], 
):
    logging.info('Filling replay with demos...')
    demo_augmentation = buffer_config['demos']['augmentation']
    demo_augmentation_every_n = buffer_config['demos']['augmentation_every_n']
    all_actions = []
    num_demos_per_variation = buffer_config['demos']['num_init_per_variation']
    assert num_demos_per_variation <= env.num_demos_per_variation, 'Requested too many initial demos.'
    demos = env.get_demos(index=0, amount=num_demos_per_variation)
    for demo in demos:
        episode_keypoints = keypoint_discovery(demo)

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue
            for ob in demo: # TODO this is my code, I should use extract_demo somewhere instead       
                ob.task_low_dim_state = None 
            obs = demo[i]
            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            all_actions.extend(_add_keypoints_to_replay(
                cameras, replay, obs, demo, env, episode_keypoints))
    logging.info('Replay filled with demos.')
    return all_actions
