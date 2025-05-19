import numpy as np
import json
import os
from enum import Enum
import scipy.stats
import torch
import  utils

class MediapipeOptions(Enum):
    XYC = 1
    XYZ = 2
    XYZC = 3


class GenFeaturesMediapipeC4():
    def __init__(self, cfg, option, normalize=False, offset = False, noFramesLimit = False):
        print ('GenFeaturesMediapipeC4')
        print ('Number joints: {}'.format(cfg['NUM_KPS']))
        print ('Option: {}'.format(option))
        
        self.option = option
        self.normalize = normalize
        self.offset = offset
        self.noFramesLimit = noFramesLimit
        self.mean = None
        self.std = None
        self.cfg = cfg
        
        print('normalize: {}'.format(self.normalize))
        print('offset: {}'.format(self.offset))
        print('noFramesLimit: {}'.format(self.noFramesLimit))

    def getFeatures(self, data, mean=None, std=None):
        channels = self.cfg['NUM_CHANNELS']
        number_frames_video = data.shape[0]
        if (self.noFramesLimit == False):
            data = self.cut_kps_array_from_middle(data, number_frames_video, self.cfg['MAX_FRAMES'])
            number_frames_video = self.cfg['MAX_FRAMES']

        if (self.offset):
            data = self.apply_offset_hands(data)
        if (self.normalize):
            data = self.apply_normalize(data, self.option)

        data_joints = np.zeros((channels, number_frames_video, self.cfg['NUM_KPS'], self.cfg['NUM_PERSON']))
        for idx, frame in enumerate(data):
            pose = [v for i, v in enumerate(frame.flatten()) if i % 4 != 3 and i // 4 in self.cfg['KEYPOINTS_INCLUDE']]    
            score = [v for i, v in enumerate(frame.flatten()) if i % 4 == 3 and i // 4 in self.cfg['KEYPOINTS_INCLUDE']]

            for m in range(self.cfg['NUM_PERSON']):
                data_joints[0, idx, :, m] = pose[0::3]
                data_joints[1, idx, :, m] = pose[1::3]
                data_joints[2, idx, :, m] = pose[2::3]
                data_joints[3, idx, :, m] = score
        
        data_bones = self.compute_bones(data_joints)    

        data_motion_joints = self.compute_motion_average(data_joints, number_frames_video, frames_used=1)
        # data_motion_joints5 = self.compute_motion_average(data_joints, number_frames_video, frames_used=5)
        data_motion_joints5 = []
        data_motion_bones = self.compute_motion_average(data_bones, number_frames_video, frames_used=1)
        data_motion_bones5 = []
        # data_motion_bones5 = self.compute_motion_average(data_bones, number_frames_video, frames_used=1)
        
        # data_angles = self.compute_angles(data_joints, self.option)
        data_angles, data_angles_extended = self.compute_angles_extended(data_joints, self.option)
        # data_angles_center = self.compute_angles_center(data_joints, self.option)
        data_angles_center = []

        # print('data_joints: ', data_joints.shape)
        # print('data_bones: ', data_bones.shape)
        # print('data_angles: ', data_angles.shape)
        # print('data_angles_extended: ', data_angles_extended.shape)

        if (self.option==MediapipeOptions.XYC):
            data_joints = data_joints[[0,1,3],:,:,:]
            data_bones = data_bones[[0,1,3],:,:,:]
            data_motion_joints = data_motion_joints[[0,1,3],:,:,:]
            data_motion_bones = data_motion_bones[[0,1,3],:,:,:]
            # data_motion_joints5 = data_motion_joints5[[0,1,3],:,:,:]
            # data_motion_bones5 = data_motion_bones5[[0,1,3],:,:,:]
        elif (self.option==MediapipeOptions.XYZ):
            data_joints = data_joints[[0,1,2],:,:,:]
            data_bones = data_bones[[0,1,2],:,:,:]
            data_motion_joints = data_motion_joints[[0,1,2],:,:,:]
            data_motion_bones = data_motion_bones[[0,1,2],:,:,:]
            # data_motion_joints5 = data_motion_joints5[[0,1,2],:,:,:]
            # data_motion_bones5 = data_motion_bones5[[0,1,2],:,:,:]
            
        return data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_extended, data_angles_center, data
    
    def saveFeatures(self, type_kps, folder_out, filename, data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_extended, data_angles_center, data):

        ###  INDIVIDUAL
        #### no_angles
        folder_out_joints = os.path.join(folder_out, 'joints_'+type_kps)
        folder_out_bones = os.path.join(folder_out, 'bones_'+type_kps)
        folder_out_joints_motion = os.path.join(folder_out, 'joints_motion_'+type_kps)
        folder_out_bones_motion = os.path.join(folder_out, 'bones_motion_'+type_kps)
        folder_out_joints_motion5 = os.path.join(folder_out, 'joints_motion5_'+type_kps)
        folder_out_bones_motion5 = os.path.join(folder_out, 'bones_motion5_'+type_kps)
        folder_out_angles = os.path.join(folder_out, 'angles_'+type_kps)
        folder_out_angles_extended = os.path.join(folder_out, 'angles_extended_'+type_kps)

        #### angles
        folder_out_joints_angles = folder_out_joints + 'a'
        folder_out_joints_angles_center = folder_out_joints + 'g'
        folder_out_bones_angles = folder_out_bones + 'a'
        folder_out_bones_angles_center = folder_out_bones + 'g'

        # MIX
        #### no_angles
        folder_out_joints_bones = os.path.join(folder_out, 'joints_bones_'+type_kps)
        folder_out_joints_bones_motion = os.path.join(folder_out, 'joints_bones_motion_'+type_kps)
        folder_out_joints_bones_motion5 = os.path.join(folder_out, 'joints_bones_motion5_'+type_kps)

        #### angles
        folder_out_joints_bones_angles = folder_out_joints_bones + 'a'
        folder_out_joints_bones_angles_center = folder_out_joints_bones + 'g'
        folder_out_joints_bones_motion_angles = folder_out_joints_bones_motion + 'a'
        folder_out_joints_bones_motion_angles_center = folder_out_joints_bones_motion + 'g'
        folder_out_joints_bones_motion5_angles = folder_out_joints_bones_motion5 + 'a'
        folder_out_joints_bones_motion5_angles_center = folder_out_joints_bones_motion5 + 'g'

        utils.create_folder(folder_out_joints, reset=False)
        utils.create_folder(folder_out_bones, reset=False)
        utils.create_folder(folder_out_joints_motion, reset=False)
        utils.create_folder(folder_out_bones_motion, reset=False)
        utils.create_folder(folder_out_joints_motion5, reset=False)
        utils.create_folder(folder_out_bones_motion5, reset=False)
        utils.create_folder(folder_out_angles, reset=False)
        utils.create_folder(folder_out_angles_extended, reset=False)

        utils.create_folder(folder_out_joints_angles, reset=False)
        # utils.create_folder(folder_out_joints_angles_center, reset=False)
        utils.create_folder(folder_out_bones_angles, reset=False)
        # utils.create_folder(folder_out_bones_angles_center, reset=False)
        # utils.create_folder(folder_out_joints_bones, reset=False)
        # utils.create_folder(folder_out_joints_bones_motion, auto=True)
        # utils.create_folder(folder_out_joints_bones_motion5, auto=True)
        # utils.create_folder(folder_out_joints_bones_angles, reset=False)
        # utils.create_folder(folder_out_joints_bones_angles_center, reset=False)
        # utils.create_folder(folder_out_joints_bones_motion_angles, reset=False)
        # utils.create_folder(folder_out_joints_bones_motion_angles_center, reset=False)
        # utils.create_folder(folder_out_joints_bones_motion5_angles, reset=False)
        # utils.create_folder(folder_out_joints_bones_motion5_angles_center, reset=False)

        folder_out_data = os.path.join(folder_out, 'data_'+type_kps)
        utils.create_folder(folder_out_data, reset=False)


        # PREPARE DATA ANGLES + JOINTS+BONES MOTION
        data_angles = np.expand_dims(data_angles, axis=0)
        # data_angles_center = np.expand_dims(data_angles_center, axis=0)
        # data_motion_joints_motion_bones = np.concatenate((data_motion_joints[:-1], data_motion_bones), axis=0)
        # data_motion_joints_motion_bones5 = np.concatenate((data_motion_joints5[:-1], data_motion_bones5), axis=0)

            
        out_path = os.path.join(folder_out_data, filename+'.npy')
        np.save(out_path, data)

        # SAVE JOINTS & BONES
        out_path = os.path.join(folder_out_joints, filename+'.npy')
        np.save(out_path, data_joints)
        out_path = os.path.join(folder_out_bones, filename+'.npy')
        np.save(out_path, data_bones)

        # SAVE JOINTS & BONES MOTION
        out_path = os.path.join(folder_out_joints_motion, filename+'.npy')
        np.save(out_path, data_motion_joints)
        out_path = os.path.join(folder_out_bones_motion, filename+'.npy')
        np.save(out_path, data_motion_bones)

        # # SAVE JOINTS & BONES MOTION - 5
        # out_path = os.path.join(folder_out_joints_motion5, filename+'.npy')
        # np.save(out_path, data_motion_joints5)
        # out_path = os.path.join(folder_out_bones_motion5, filename+'.npy')
        # np.save(out_path, data_motion_bones5)

        out_path = os.path.join(folder_out_angles, filename+'.npy')
        np.save(out_path, data_angles)

        # SAVE ANGLES WITH CONFIDENCES
        out_path = os.path.join(folder_out_angles_extended, filename+'.npy')
        np.save(out_path, data_angles_extended)




        # SAVE JOINTS + ANGLE
        data_joints_angles = np.concatenate((data_joints, data_angles), axis=0)
        out_path = os.path.join(folder_out_joints_angles, filename+'.npy')
        np.save(out_path, data_joints_angles)

        # SAVE BONE + ANGLE
        data_bones_angles = np.concatenate((data_bones, data_angles), axis=0)
        out_path = os.path.join(folder_out_bones_angles, filename+'.npy')
        np.save(out_path, data_bones_angles)

        # # SAVE JOINTS + ANGLE_CENTER
        # data_joints_angles_center = np.concatenate((data_joints, data_angles_center), axis=0)
        # out_path = os.path.join(folder_out_joints_angles_center, filename+'.npy')
        # np.save(out_path, data_joints_angles_center)

        # # SAVE BONE + ANGLE_CENTER
        # data_bones_angles_center = np.concatenate((data_bones, data_angles_center), axis=0)
        # out_path = os.path.join(folder_out_bones_angles_center, filename+'.npy')
        # np.save(out_path, data_bones_angles_center)

        # # JOINTS + BONES
        # data_joints_bones = np.concatenate((data_joints[:-1], data_bones), axis=0)
        # out_path = os.path.join(folder_out_joints_bones, filename+'.npy')
        # np.save(out_path, data_joints_bones)


        # # JOINTS + BONES + ANGLE
        # data_joints_bones_angles = np.concatenate((data_joints_bones, data_angles), axis=0)
        # out_path = os.path.join(folder_out_joints_bones_angles, filename+'.npy')
        # np.save(out_path, data_joints_bones_angles)

        # # JOINTS + BONES + ANGLE_CENTER
        # data_joints_bones_angles_center = np.concatenate((data_joints_bones, data_angles_center), axis=0)
        # out_path = os.path.join(folder_out_joints_bones_angles_center, filename+'.npy')
        # np.save(out_path, data_joints_bones_angles_center)


        # # JOINTS + BONES + MOTION + ANGLE
        # data_joints_bones_joints_motion = np.concatenate((data_joints_bones[:-1], data_motion_joints_motion_bones), axis=0)
        # data_joints_bones_joints_motion_angle = np.concatenate((data_joints_bones_joints_motion, data_angles), axis=0)
        # out_path = os.path.join(folder_out_joints_bones_motion_angles, filename+'.npy')
        # np.save(out_path, data_joints_bones_joints_motion_angle)


        # # JOINTS + BONES + MOTION + ANGLE_CENTER
        # data_joints_bones_joints_motion_angle_center = np.concatenate((data_joints_bones_joints_motion, data_angles_center), axis=0)
        # out_path = os.path.join(folder_out_joints_bones_motion_angles_center, filename+'.npy')
        # np.save(out_path, data_joints_bones_joints_motion_angle_center)


        # # JOINTS + BONES + MOTION5 + ANGLE
        # data_joints_bones_joints_motion5 = np.concatenate((data_joints_bones[:-1], data_motion_joints_motion_bones5), axis=0)
        # data_joints_bones_joints_motion5_angle = np.concatenate((data_joints_bones_joints_motion5, data_angles), axis=0)
        # out_path = os.path.join(folder_out_joints_bones_motion5_angles, filename+'.npy')
        # np.save(out_path, data_joints_bones_joints_motion5_angle)


        # # JOINTS + BONES + MOTION5 + ANGLE_CENTER
        # data_joints_bones_joints_motion5_angle_center = np.concatenate((data_joints_bones_joints_motion5, data_angles_center), axis=0)
        # out_path = os.path.join(folder_out_joints_bones_motion5_angles_center, filename+'.npy')
        # np.save(out_path, data_joints_bones_joints_motion5_angle_center)

    
    def calculate_init_frame(self, number_frames_video, size_max, random_center):
        lower = int(size_max/2)
        upper = int(number_frames_video - size_max/2)
        if (random_center == True):
            mu = ((upper - lower) / 2) + lower
            sigma = (upper - lower) / 20
            center_idx = int(scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=1))
            return int(center_idx)
        else:
            return int(((upper - lower) / 2) + lower)
    
    def remove_frames_random(self, data):
        return data[np.random.rand(len(data)) > 0.1,:,:]

    def cut_kps_array_from_middle(self, data, number_frames_video, max_frame, random_center=False):
        if (number_frames_video > max_frame* 2):
            data = self.remove_frames_random(data)
            number_frames_video = data.shape[0]
            
        if number_frames_video > max_frame:
            frame_init = self.calculate_init_frame(number_frames_video, max_frame, random_center)
            frame_end = frame_init + max_frame
            data = data[frame_init:frame_end,:,:]
        return data


    ### GEN FEATURES
    ### COMPUTE BONES
    # def compute_bones(self, data_joints):
    #     data_bones = np.zeros_like(data_joints)
    #     for v1, v2 in self.cfg['CONNECTIONS']:
    #         # v1 -= 1
    #         # v2 -= 1
    #         data_bones[:3, :, v1, :] = data_joints[:3, :, v1, :] - data_joints[:3, :, v2, :]
    #         data_bones[3, :, v1, :] = (data_joints[3, :, v1, :] + data_joints[3, :, v2, :]) / 2
    #     return data_bones

    def compute_bones(self, data_joints):
        data_bones = np.zeros_like(data_joints)
        for idx, (v1, v2) in enumerate(self.cfg['CONNECTIONS']):
            data_bones[:3, :, idx, :] = data_joints[:3, :, v1, :] - data_joints[:3, :, v2, :]
            data_bones[3, :, idx, :] = (data_joints[3, :, v1, :] + data_joints[3, :, v2, :]) / 2
        return data_bones
        

    ### COMPUTE MOTION
    def compute_motion(self, data, number_frames_video):
        data_motion = np.zeros_like(data)
        for t in range(number_frames_video - 1):
            data_motion[:3, t, :, :] = data[:3, t + 1, :, :] - data[:3, t, :, :]
            data_motion[3, t, :, :] = ( data[3, t + 1, :, :] + data[3, t, :, :] ) /2
        data_motion[:3, number_frames_video - 1, :, :] = 0
        return data_motion


    def compute_motion_average(self, data, number_frames_video, frames_used=1):
        # Asegurándose de que X no sea mayor que el número de frames en el video
        frames_used = min(frames_used, number_frames_video - 1)
        # Inicializar data_motion con ceros
        data_motion = np.zeros_like(data)
        
        for t in range(number_frames_video):
            # Para los últimos X frames, no hay suficientes frames posteriores para promediar
            if t >= number_frames_video - frames_used:
                # Aquí simplemente podrías dejar el movimiento calculado como 0 para estos frames
                data_motion[:3, t, :, :] = 0
            else:
                # Calcular el promedio de los X frames posteriores
                data_motion[:3, t, :, :] = np.mean(data[:3, t+1:t+frames_used+1, :, :], axis=1) - data[:3, t, :, :]
        
        return data_motion


    def compute_angles(self, data, option, useRadians=False):

        channels = 3  # x y z
        if (option==MediapipeOptions.XYC):
            channels = 2  # x y

        angles = np.array(self.cfg['ANGLES'])
        x = torch.from_numpy(np.transpose(np.squeeze(data, axis=-1), (1, 2, 0))[..., :]).float()
        the_joint = x[:, angles[:, 0], :channels]
        v1 = x[:, angles[:, 1], :channels]
        v2 = x[:, angles[:, 2], :channels]
        vec1 = v1 - the_joint
        vec2 = v2 - the_joint
        angle_data = torch.nn.functional.cosine_similarity(vec1, vec2, 2, 0.0)
        angle_data = torch.nan_to_num(angle_data, nan=1.0)  # Reemplaza NaN con 1.0 - 1.0 = total similitud (0 grados)

        if (useRadians):
            clamped_cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
            angle_data = torch.acos(clamped_cosine_similarity)

        angle_data = angle_data.unsqueeze(2)
        # angle_data = torch.where(torch.isnan(angle_data), torch.tensor(0.0), angle_data)
        angle_data_np = angle_data.numpy()
        angle_data_np = np.transpose(angle_data_np,(2,0,1))
        angle_data_np = np.expand_dims(angle_data_np, axis=-1)
        return angle_data.numpy()


    # def compute_angles_extended(self, data, option, useRadians=False):
    #     # print('compute_angles_extended')

    #     channels = 3  # x y z
    #     if (option == MediapipeOptions.XYC):
    #         channels = 2  # x y

    #     angles = np.array(self.cfg['ANGLES'])
    #     x = torch.from_numpy(np.transpose(np.squeeze(data, axis=-1), (1, 2, 0))[..., :]).float()
    #     the_joint = x[:, angles[:, 0], :channels]
    #     v1 = x[:, angles[:, 1], :channels]
    #     v2 = x[:, angles[:, 2], :channels]
    #     cv1 = x[:, angles[:, 1], 3]
    #     cv2 = x[:, angles[:, 2], 3]
    #     vec1 = v1 - the_joint
    #     vec2 = v2 - the_joint
    #     angle_data = torch.nn.functional.cosine_similarity(vec1, vec2, 2, 0.0)
    #     angle_data = torch.nan_to_num(angle_data, nan=1.0)  # Replace NaN with 1.0 - 1.0 = total similarity (0 degrees)

    #     if useRadians:
    #         clamped_cosine_similarity = torch.clamp(angle_data, -1.0, 1.0)
    #         angle_data = torch.acos(clamped_cosine_similarity)

    #     angle_data = angle_data.unsqueeze(2)
    #     angle_data_np = angle_data.numpy()
    #     angle_data_np = np.transpose(angle_data_np, (2, 0, 1))
    #     angle_data_np = np.expand_dims(angle_data_np, axis=-1)

    #     # print('angle_data_np shape: ', angle_data_np.shape)  # Should be (1, 100, 61, 1)
    #     # print('cv1 shape: ', cv1.shape)  # Should be (100, 61)
    #     # print('cv2 shape: ', cv2.shape)  # Should be (100, 61)

    #     # Initialize the results array
    #     results = np.zeros((3, 100, 61, 1))

    #     # Assign angle_data_np to the first slice
    #     results[0, :, :, :] = angle_data_np

    #     # Expand dimensions of cv1 and cv2
    #     cv1_expanded = np.expand_dims(cv1.numpy(), axis=-1)
    #     cv2_expanded = np.expand_dims(cv2.numpy(), axis=-1)

    #     # Assign cv1 and cv2 to the second and third slices
    #     results[1, :, :, :] = cv1_expanded
    #     results[2, :, :, :] = cv2_expanded

    #     return results


    def compute_angles_extended(self, data, option, useRadians=False):
        # print('compute_angles_extended')

        channels = 3  # x y z
        if (option == MediapipeOptions.XYC):
            channels = 2  # x y

        angles = np.array(self.cfg['ANGLES'])
        x = torch.from_numpy(np.transpose(np.squeeze(data, axis=-1), (1, 2, 0))[..., :]).float()
        the_joint = x[:, angles[:, 0], :channels]
        v1 = x[:, angles[:, 1], :channels]
        v2 = x[:, angles[:, 2], :channels]
        cvj = x[:, angles[:, 0], 3]
        cv1 = x[:, angles[:, 1], 3]
        cv2 = x[:, angles[:, 2], 3]
        vec1 = v1 - the_joint
        vec2 = v2 - the_joint
        angle_data = torch.nn.functional.cosine_similarity(vec1, vec2, 2, 0.0)
        angle_data = torch.nan_to_num(angle_data, nan=1.0)  # Replace NaN with 1.0 - 1.0 = total similarity (0 degrees)

        if useRadians:
            clamped_cosine_similarity = torch.clamp(angle_data, -1.0, 1.0)
            angle_data = torch.acos(clamped_cosine_similarity)

        angle_data = angle_data.unsqueeze(2)
        angle_data_np = angle_data.numpy()
        angle_data_np = np.transpose(angle_data_np, (2, 0, 1))
        angle_data_np = np.expand_dims(angle_data_np, axis=-1)

        # print('angle_data_np shape: ', angle_data_np.shape)  # Should be (1, 100, 61, 1)
        # print('cv1 shape: ', cv1.shape)  # Should be (100, 61)
        # print('cv2 shape: ', cv2.shape)  # Should be (100, 61)

        # Initialize the results array
        angle_data_extended = np.zeros((4, 100, 61, 1))

        # Assign angle_data_np to the first slice
        angle_data_extended[0, :, :, :] = angle_data_np

        # Expand dimensions of cv1 and cv2
        cvj_expanded = np.expand_dims(cv1.numpy(), axis=-1)
        cv1_expanded = np.expand_dims(cv1.numpy(), axis=-1)
        cv2_expanded = np.expand_dims(cv2.numpy(), axis=-1)

        # Assign cv1 and cv2 to the second and third slices
        angle_data_extended[1, :, :, :] = cvj_expanded
        angle_data_extended[2, :, :, :] = cv1_expanded
        angle_data_extended[3, :, :, :] = cv2_expanded

        return angle_data.numpy(), angle_data_extended



    def compute_angles_center(self, data, option, useRadians=False):
        
        channels = 3  # x y z
        if (option==MediapipeOptions.XYC):
            channels = 2  # x y



        angles = np.array(self.cfg['ANGLES'])
        x = torch.from_numpy(np.transpose(np.squeeze(data, axis=-1), (1, 2, 0))[..., :3]).float()
        v1 = x[:, [0]*self.cfg['NUM_KPS'], :channels]                             # always key point 0
        v2 = x[:, range(self.cfg['NUM_KPS']), :channels]

        vec1 = v1 # - center_zero
        vec2 = v2 # - center_zero
        angle_data = torch.nn.functional.cosine_similarity(vec1, vec2, 2, 0.0)
        angle_data = torch.nan_to_num(angle_data, nan=1.0)  # Reemplaza NaN con 1.0 - 1.0 = total similitud (0 grados)

        if (useRadians):
            clamped_cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
            angle_data = torch.acos(clamped_cosine_similarity)

        angle_data = angle_data.unsqueeze(2)
        angle_data_np = angle_data.numpy()
        angle_data_np = np.transpose(angle_data_np,(2,0,1))
        angle_data_np = np.expand_dims(angle_data_np, axis=-1)
        return angle_data.numpy()


    def apply_normalize(self, data, option):
        data_norm = data.copy()

        channels = 3  # x y z
        if (option==MediapipeOptions.XYC):
            channels = 2  # x y
            # print('used norm 2D XY - 2 channels')
            
        lshoulder = data[:, self.cfg['POSE_POSITION_SHOULDER_LEFT'], 0:channels]
        rshoulder = data[:, self.cfg['POSE_POSITION_SHOULDER_RIGHT'], 0:channels]
        ref = (rshoulder + lshoulder) / 2


        ref = np.expand_dims(ref, axis=1)
        data_norm[:, :, 0:channels] = data[:, :, 0:channels] - ref
        distance = self.averageDistanceBetweenShoulders(data, channels)

        # Usamos este valor promedio para la normalización
        data_norm[:, :, 0:channels] = np.divide(data_norm[:, :, 0:channels], distance)
        data_norm[:, :, 0:channels] = np.nan_to_num(data_norm[:, :, 0:channels], nan=0.0)

        return data_norm

    def averageDistanceBetweenShoulders(self, data, channels):
        rshoulder = data[:, self.cfg['POSE_POSITION_SHOULDER_LEFT'], 0:channels]
        lshoulder = data[:, self.cfg['POSE_POSITION_SHOULDER_RIGHT'], 0:channels]
        distances = np.linalg.norm(lshoulder - rshoulder, axis=1)
        
        # Calcula la distancia promedio a lo largo de todos los frames
        average_distance = np.mean(distances)
        return average_distance

    def apply_offset_hands(self, data):
        data_change = np.copy(data)
        for i_frame in range(data.shape[0]):
            offset_XYZ_hand_left = data[i_frame,self.cfg['POSE_POSITION_WRIST_LEFT'],0:3] - data[i_frame, self.cfg['HANDS_POSITION_WRIST_LEFT'],0:3]
            offset_XYZ_hand_right = data[i_frame, self.cfg['POSE_POSITION_WRIST_RIGHT'],0:3] - data[i_frame, self.cfg['HANDS_POSITION_WRIST_RIGHT'],0:3]
            data_change[i_frame,self.cfg['HANDS_POSITION_WRIST_LEFT']:(self.cfg['HANDS_POSITION_WRIST_LEFT_LAST_IDX']+1),0:3] = data_change[i_frame,self.cfg['HANDS_POSITION_WRIST_LEFT']:(self.cfg['HANDS_POSITION_WRIST_LEFT_LAST_IDX']+1),0:3] + offset_XYZ_hand_left
            data_change[i_frame,self.cfg['HANDS_POSITION_WRIST_RIGHT']:(self.cfg['HANDS_POSITION_WRIST_RIGHT_LAST_IDX']+1),0:3] = data_change[i_frame,self.cfg['HANDS_POSITION_WRIST_RIGHT']:(self.cfg['HANDS_POSITION_WRIST_RIGHT_LAST_IDX']+1),0:3] + offset_XYZ_hand_right
        return data_change
        


