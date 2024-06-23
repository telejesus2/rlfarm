import logging
import random
from typing import Dict, Tuple
import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import pickle
from PIL import Image

from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.backend.observation import Observation
from pyrep.objects.object import Object
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from rlfarm.envs.rlbench.rlbench_env import RLBenchEnv
from rlfarm.utils.transition import Transition
from rlfarm.envs.rlbench.builder import AVAILABLE_TASKS
from rlfarm.vision.observation import Observation as _Observation


FORMAT = 'vision'  # one of ['demo', 'vision']
IMG_SIZE = (128, 128)

# utilites for FORMAT = 'vision'
SAVE_IMAGES = True

# utilites for FORMAT = 'demo'
from rlbench.backend import utils
from rlbench.backend.const import *
from pyrep.const import RenderMode
# same as in RLBench/tools/dataset_generator.py
def save_demo(demo, example_path):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)
    print(front_mask_path)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


class RecorderRLBenchEnv(RLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(RecorderRLBenchEnv, self).__init__(*args, **kwargs)

        if FORMAT == 'vision':

            cam_config = CameraConfig(image_size=IMG_SIZE)
            obs_config = ObservationConfig(cam_config, 
                copy.deepcopy(cam_config), copy.deepcopy(cam_config),
                copy.deepcopy(cam_config), copy.deepcopy(cam_config))
            obs_config.joint_velocities = True
            obs_config.joint_positions = True
            obs_config.gripper_open = True
            obs_config.gripper_pose = True
            obs_config.task_low_dim_state = True

            if SAVE_IMAGES:
                # front camera
                obs_config.front_camera.rgb = True
                obs_config.front_camera.mask = True
                obs_config.front_camera.depth = True
                obs_config.front_camera.depth_in_meters = False
                obs_config.front_camera.masks_as_one_channel = True
                # wrist camera
                obs_config.wrist_camera.rgb = True
                obs_config.wrist_camera.mask = True
                obs_config.wrist_camera.depth = True
                obs_config.wrist_camera.depth_in_meters = False
                obs_config.wrist_camera.masks_as_one_channel = True

        elif FORMAT == 'demo':

            img_size = IMG_SIZE
            obs_config = ObservationConfig()
            obs_config.set_all(True)
            obs_config.gripper_touch_forces = False
            obs_config.right_shoulder_camera.image_size = img_size
            obs_config.left_shoulder_camera.image_size = img_size
            obs_config.overhead_camera.image_size = img_size
            obs_config.wrist_camera.image_size = img_size
            obs_config.front_camera.image_size = img_size

            # Store depth as 0 - 1
            obs_config.right_shoulder_camera.depth_in_meters = False
            obs_config.left_shoulder_camera.depth_in_meters = False
            obs_config.overhead_camera.depth_in_meters = False
            obs_config.wrist_camera.depth_in_meters = False
            obs_config.front_camera.depth_in_meters = False

            # We want to save the masks as rgb encodings.
            obs_config.left_shoulder_camera.masks_as_one_channel = False
            obs_config.right_shoulder_camera.masks_as_one_channel = False
            obs_config.overhead_camera.masks_as_one_channel = False
            obs_config.wrist_camera.masks_as_one_channel = False
            obs_config.front_camera.masks_as_one_channel = False

            obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
            obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
            obs_config.overhead_camera.render_mode = RenderMode.OPENGL
            obs_config.wrist_camera.render_mode = RenderMode.OPENGL
            obs_config.front_camera.render_mode = RenderMode.OPENGL

        else:
            raise ValueError

        self._env._obs_config = obs_config

        self.saver = EpisodeSaver(self.max_episode_steps)

    def step(self, action: np.ndarray) -> Transition:
        obs = self._previous_obs_dict  # in case action fails.
        try:
            planning = False # TODO
            if planning:
                if self._task_name == "pick_and_place":
                    cube_x, cube_y, cube_angle, target_x, target_y = obs['low_dim_state'][20:25]
                    action_list = []
                    action_list.append(np.array([1, 0, cube_x, cube_y, cube_angle])) # grasp
                    action_list.append(np.array([0, 1, target_x, target_y, 0])) # release
                    action = action_list[self._i]

            obs, reward, terminate, info = self._task.step(action)
            self.saver.step(obs, action, reward, terminate, info) # new wrt parent
            obs = self._extract_ob(obs)
            self._previous_obs_dict = obs
        # except (IKError, ConfigurationPathError, InvalidActionError) as e:
        except: # new wrt parent
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
        self.saver.reset(obs) # new wrt parent
        del descriptions  # Not used.
        self._i = 0
        self._prev_action = None
        # reset summaries
        self.episode_summaries = [] # see runners.samplers.rollout_generator.RolloutGenerator

        # track previous obs (in case action fails in step())
        self._previous_obs_dict = self._extract_ob(obs)

        return self._previous_obs_dict

    def _extract_ob(self, ob: Observation, variation=None, t=None, prev_action=None,
                    include_low_dim=True, include_rgb=True) -> Dict[str, np.ndarray]:

        # return super()._extract_ob(ob, variation=variation, t=t, prev_action=prev_action,
        #             include_low_dim=include_low_dim, include_rgb=False) 
        
        ob_dict = vars(ob)
        new_ob = {}
        if variation is None:
            variation = self._variation

        # add low dim state
        if include_low_dim:
            key = 'low_dim_state'
            if FORMAT == 'vision':
                low_dim_state = np.array(ob.get_low_dim_data(), dtype=np.float32)
            elif FORMAT == 'demo':
                low_dim_data = [[ob.gripper_open]]
                for data in [ob.joint_velocities, ob.joint_positions,
                            ob.gripper_pose, ob.task_low_dim_state]:
                    low_dim_data.append(data)
                low_dim_state =  np.array(np.concatenate(low_dim_data), dtype=np.float32)
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
        # if ob_config.joint_forces:               low_dim_state_len += 6
        if ob_config.gripper_open:               low_dim_state_len += 1
        if ob_config.gripper_pose:               low_dim_state_len += 7
        # if ob_config.gripper_joint_positions:    low_dim_state_len += 2
        # if ob_config.gripper_touch_forces:       low_dim_state_len += 2
        if ob_config.task_low_dim_state:         low_dim_state_len += AVAILABLE_TASKS[self._task_name][1]
        if self._state_includes_variation_index: low_dim_state_len += 1 
        if self._state_includes_remaining_time:  low_dim_state_len += 1
        if self._state_includes_previous_action: low_dim_state_len += self.action_shape
        ob_shape[key], ob_dtype[key] = (low_dim_state_len,), np.float32

        return ob_shape, ob_dtype

#============================================================================================#
# UTILITIES
#============================================================================================#

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def lstq(A, Y, lamb=0.01):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        print (torch.matrix_rank(A))
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = lstq(A_dash, Y_dash)
        return x
    
def least_square_normal_regress(x_depth3d, size=9, gamma=0.15, depth_scaling_factor=1, eps=1e-5):    
    stride=1

    # xyz_perm = xyz.permute([0, 2, 3, 1])
    xyz_padded = F.pad(x_depth3d, (size//2, size//2, size//2, size//2), mode='replicate')
    xyz_patches = xyz_padded.unfold(2, size, stride).unfold(3, size, stride) # [batch_size, 3, width, height, size, size]
    xyz_patches = xyz_patches.reshape((*xyz_patches.shape[:-2], ) + (-1,))  # [batch_size, 3, width, height, size*size]
    xyz_perm = xyz_patches.permute([0, 2, 3, 4, 1])

    diffs = xyz_perm - xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs = diffs / xyz_perm[:, :, :, (size*size)//2].unsqueeze(3)
    diffs[..., 0] = diffs[..., 2]
    diffs[..., 1] = diffs[..., 2]
    xyz_perm[torch.abs(diffs) > gamma] = 0.0

    A_valid = xyz_perm * depth_scaling_factor                           # [batch_size, width, height, size, 3]

    # Manual pseudoinverse
    A_trans = xyz_perm.permute([0, 1, 2, 4, 3]) * depth_scaling_factor  # [batch_size, width, height, 3, size]

    A = torch.matmul(A_trans, A_valid)

    A_det = torch.det(A)
    A[A_det < eps, :, :] = torch.eye(3, device="cpu")
    
    A_inv = torch.inverse(A)
    b = torch.ones(list(A_valid.shape[:4]) + [1], device=x_depth3d.device)
    lstsq = A_inv.matmul(A_trans).matmul(b)

    lstsq = lstsq / torch.norm(lstsq, dim=3).unsqueeze(3)
    lstsq[lstsq != lstsq] = 0.0
    return -lstsq.squeeze(-1).permute([0, 3, 1, 2])

def reproject_depth(depth, field_of_view, cached_cr=None, max_depth=1.):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.
    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    """


    dx, dy = torch.tensor(depth.shape[2:4]) - 1
    cx, cy = torch.tensor([dx, dy]) / 2

    fx, fy = torch.tensor([[depth.shape[2]], [depth.shape[3]]], device=field_of_view.device, dtype=torch.float32) \
                / (2. * torch.tan(field_of_view.float() / 2.).unsqueeze(0))

    if cached_cr is None:
        cols, rows = depth.shape[2], depth.shape[3]
        c, r = torch.tensor(np.meshgrid(np.arange(cols), np.arange(rows), sparse=False), device=field_of_view.device, dtype=torch.float32)
    else:
        c, r = cached_cr

    z = depth.squeeze(1) * max_depth

    c = c.float()
    cx = cx.float()
    cy = cy.float()
    x = z * ((c - cx).unsqueeze(0) / fx.unsqueeze(1).unsqueeze(1))
    y = z * ((r - cy).unsqueeze(0) / fy.unsqueeze(1).unsqueeze(1))
    return torch.stack((x, y, z), dim=1), cached_cr

class LeastSquareModule(torch.nn.Module):

    def __init__(self, gamma=0.15, beta=9):
        self.cached_cr = None
        self.shape = None
        self.patch_size = beta
        self.z_depth_thresh = gamma
        super().__init__()

    
    def forward(self, x_depth, field_of_view_rads):
        x_depth3d, cached_cr = reproject_depth(x_depth, field_of_view_rads, cached_cr=self.cached_cr, max_depth=1.)
        # plt.imshow(x_depth3d[0].permute(1,2,0).cpu())
        if self.cached_cr is None:
            self.cached_cr = cached_cr
        return least_square_normal_regress(x_depth3d, size=self.patch_size, gamma=self.z_depth_thresh)

#============================================================================================#
# SAVER
#============================================================================================#

class EpisodeSaver(object):
    """
    Save the experience data from a gym env to a file
    and notify the srl server so it learns from the gathered data
    :param name: (str)
    :param max_dist: (float)
    :param state_dim: (int)
    :param globals_: (dict) Environments globals
    :param learn_every: (int)
    :param learn_states: (bool)
    :param path: (str)
    :param relative_pos: (bool)
    """

    def __init__(self, max_episode_len: int, path='data/', name='slide4'):
        super(EpisodeSaver, self).__init__()
        self._max_episode_len = max_episode_len
        self.data_folder = path + name
        self.path = path
        check_and_make(self.data_folder)

        self.actions = []
        self.rewards = []
        self.states = []
        self.episode_step = 0
        self.episode_idx = -1
        self.episode_folder = None     

    def _save_image(self, observation):
        def save_image(observation, prefix="slide"):
            ob_dict = vars(observation)
            image, depth, mask = ob_dict[prefix + '_rgb'], ob_dict[prefix + '_depth'], ob_dict[prefix + '_mask']
            image_path = "{}/{}/{}/frame{:06d}".format(self.data_folder, self.episode_folder, prefix, self.episode_step)

            def save(img, type):
                img = Image.fromarray(img)
                img.save(fp="{}_{}.png".format(image_path, type))
            
            #Base Image
            rgb = (image).astype(np.uint8)
            save(rgb, "rgb")

            #Depth
            img_depth = (depth * 255).astype(np.uint8)
            save(img_depth, "depth")

            #Normals estimation
            depth = depth[ np.newaxis, np.newaxis,...]
            depth = (depth - np.min(depth))/np.max(depth)
            depth = torch.from_numpy(depth.copy()).float()#.cuda()
            fov = torch.tensor(60.0 * 3.1415926/180).float()#.cuda()
            normals = LeastSquareModule(2,3)(depth*1000, fov) 
            normals = np.array((normals[0].cpu() + 1 )/2)
            normals = (normals.transpose(1,2,0)*255).astype(np.uint8)
            save(normals, "normal")

            #Sobel Edges
            def sobel_transform(x, blur=0):
                image = x.mean(axis=0)
                blur = ndimage.filters.gaussian_filter(image, sigma=blur, )
                sx = ndimage.sobel(blur, axis=0, mode='constant')
                sy = ndimage.sobel(blur, axis=1, mode='constant')
                sob = np.hypot(sx, sy)
                # edge = torch.FloatTensor(sob).unsqueeze(0)
                return sob
            sobel = sobel_transform(image.transpose(2,0,1)/255., 0)
            sobel = np.array(sobel/np.max(sobel) * 255).astype(np.uint8)
            save(sobel, "sobel")

            #Sobel Edges 3d
            def sobel_transform_3d(x, blur=0):
                # image = x.mean(axis=0)
                blur = ndimage.filters.gaussian_filter(x, sigma=blur, )
                sx = ndimage.sobel(blur, axis=0, mode='constant')
                sy = ndimage.sobel(blur, axis=1, mode='constant')
                sob = np.hypot(sx, sy)
                # edge = torch.FloatTensor(sob).unsqueeze(0)
                return sob
            sobel = sobel_transform_3d(np.array(depth.cpu())[0][0], 0)
            sobel = np.array(sobel/np.max(sobel) * 255).astype(np.uint8)
            save(sobel, "sobel_3d")

            #Image Segmentation
            def map_new_classes(x):
                #x is the handle
                env_classes = ['ResizableFloor_5_25_visibleElement', 'workspace', 'diningTable_visible', 'Wall1', 'Wall2', 'Wall3', 'Wall4']
                gripper_classes = ['BaxterGripper_rightPad_visible', 'BaxterGripper_rightFinger_visible', 'BaxterGripper_leftPad_visible', 'BaxterGripper_leftFinger_visible', 'BaxterGripper_visible']
                gripper_classes += ['Robotiq_140_visible', 'logo', 'Scratch',
                'Lever1_L_visible', 'Lever2_L_visible', 'Lever3_L_visible', 'Lever4_L_visible',
                'Lever1_R_visible', 'Lever2_R_visible', 'Lever3_R_visible', 'Lever4_R_visible',]
                robot_classes = ['UR3_link7_prolong', 'UR3_link1_visible', 'UR3_link2_visible', 'UR3_link3_visible', 'UR3_link4_visible', 'UR3_link5_visible', 'UR3_link6_visible', 'UR3_link7_visible']
                classes = env_classes + robot_classes + gripper_classes
                new_inds = [0,1,1,0,0,0,0] + [2]*len(robot_classes) + [3]*len(gripper_classes)

                # mapping_task_to_scene_objects = {
                #     'push_button': {
                #         'push_button_target': 4
                #     },
                #     'reach_target_easy': {
                #         'target': 4
                #     },
                #     'slide_block_to_target': {
                #         'block': 4,
                #         'target':4
                #     },
                #     'pick_and_lift': {
                #         'pick_and_lift_target': 4,
                #         'success_visual':4
                #     }
                # }
                name = Object.get_object_name(int(x))
                if name == '':
                    return 4
                if name == 'Floor':
                    return 0
                try:
                    ind = new_inds[classes.index(name)]
                except:
                    ind = 4
                return ind
            ordered = np.vectorize(map_new_classes)(mask)
            # colores = [[0,0,0],[128,128,0],[255,128,0],[0,128,255],[255,255,255]]
            # newmask = np.zeros((128,128,3))
            # for i in range(128):
            #     for j in range(128):
            #         newmask[i,j,:] = np.array(colores[ordered[i,j]])
            # save(newmask.astype(np.uint8), 'segment_img_debug')
            save(ordered.astype(np.uint8), 'segment_img')
           
            def denoise(img, std):
                # takes in rgb, which is 128, 128, 3 shaped and 0 - 255 ints
                return np.clip(((img / 255 + np.random.normal(0,std,img.shape)) * 255), 0, 255)
            denoise = denoise(rgb,0.1).astype(np.uint8)
            save(denoise, "denoise")

        os.makedirs("{}/{}/{}".format(self.data_folder, self.episode_folder, "front"), exist_ok=True)
        os.makedirs("{}/{}/{}".format(self.data_folder, self.episode_folder, "wrist"), exist_ok=True)
        save_image(observation, "front")
        save_image(observation, "wrist")

    def reset(self, observation):
        """
        Called when starting a new episode
        :param observation: 
        :param ground_truth: (numpy array)
        """
        self.episode_idx += 1
        self.actions = []
        self.rewards = []
        self.states = []
        self.episode_step = 0
        self.episode_folder = "record_{:03d}".format(self.episode_idx)
        os.makedirs("{}/{}".format(self.data_folder, self.episode_folder), exist_ok=True)

        self.states.append(observation)
        if FORMAT == 'vision' and SAVE_IMAGES:
            self._save_image(observation)

    def step(self, observation, action, reward, done, info):
        """
        :param observation
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        :param ground_truth_state: (numpy array)
        """
        self.episode_step += 1
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(observation)
        if FORMAT == 'vision' and SAVE_IMAGES:
            self._save_image(observation)

        if done or self.episode_step == self._max_episode_len:
            if FORMAT == 'vision':
                if not info.get('success', False) and len(self.states) < 10: # TODO remove len constraint
                    self.save(done)
            elif FORMAT == 'demo':
                if info.get('success', False) and len(self.states) < 10: # TODO remove len constraint
                    for i, obs in enumerate(self.states):
                        if i > 0:
                            obs.misc['action'] = self.actions[i-1]
                            obs.misc['reward'] = self.rewards[i-1]
                    save_demo(self.states, os.path.join(self.data_folder, self.episode_folder))
            else:
                raise ValueError

    def save(self, done):
        print("Saving low dim data...")

        data = {
            'rewards': np.array(self.rewards),
            'actions': np.array(self.actions),
            'done': done,
        }
        # if len(data['actions']) != 2: # TODO remove
        #     return
        np.savez('{}/{}/episode_data.npz'.format(self.data_folder, self.episode_folder), **data)

        for i, obs in enumerate(self.states):
            # We save the images separately, so set these to None for pickling.
            obs.left_shoulder_rgb = None
            obs.left_shoulder_depth = None
            obs.left_shoulder_point_cloud = None
            obs.left_shoulder_mask = None
            obs.right_shoulder_rgb = None
            obs.right_shoulder_depth = None
            obs.right_shoulder_point_cloud = None
            obs.right_shoulder_mask = None
            obs.overhead_rgb = None
            obs.overhead_depth = None
            obs.overhead_point_cloud = None
            obs.overhead_mask = None
            obs.wrist_rgb = None
            obs.wrist_depth = None
            obs.wrist_point_cloud = None
            obs.wrist_mask = None
            obs.front_rgb = None
            obs.front_depth = None
            obs.front_point_cloud = None
            obs.front_mask = None
            if i > 0:
                obs.misc['action'] = self.actions[i-1]
                obs.misc['reward'] = self.rewards[i-1]

        # Save the low-dimension data
        with open(os.path.join(self.data_folder, self.episode_folder, 'low_dim_obs.pkl'), 'wb') as f:
            # pickle.dump(self.states, f)
            if FORMAT == 'vision':
                pickle.dump([_Observation(**vars(o)) for o in self.states], f) # to unpickle without importing rlbench
            elif FORMAT == 'demo':
                pickle.dump([Observation(**vars(o)) for o in self.states], f)