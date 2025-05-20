from api.genFeatures import GenFeaturesMediapipeC4, MediapipeOptions
import argparse
import os
from tqdm import tqdm
import yaml
import numpy as np
import shutil
import utils


def main(args, cfg):
    folder_in_kps = args.folder_in_kps
    folder_out_features = args.folder_out_features

    type_kps = args.type_kps
    offset = args.offset
    normalize = args.normalize
    noFramesLimit = args.noFramesLimit
    jump_reset = args.jump_reset
    
    print('######################################')
    print('ARGUMENTOS ENTRADA: ')
    print('folder_in_kps: ',folder_in_kps)
    print('folder_out_features: ',folder_out_features)
    print('type_kps: ',type_kps)
    print('offset: ',offset)
    print('normalize: ',normalize)
    print('noFramesLimit: ',noFramesLimit)
    print('######################################')

    if (type_kps=='C3_xyc'):
        gen_features = GenFeaturesMediapipeC4(cfg, MediapipeOptions.XYC, normalize=normalize, offset=offset, noFramesLimit=noFramesLimit)
    elif (type_kps=='C3_xyz'):
        gen_features = GenFeaturesMediapipeC4(cfg, MediapipeOptions.XYZ, normalize=normalize, offset=offset, noFramesLimit=noFramesLimit)
    elif (type_kps=='C4_xyzc'):
        gen_features = GenFeaturesMediapipeC4(cfg, MediapipeOptions.XYZC, normalize=normalize, offset=offset, noFramesLimit=noFramesLimit)

    utils.create_folder(folder_out_features, reset=jump_reset)

    arr_input_files = os.listdir(folder_in_kps)
    for file_idx in tqdm(arr_input_files):
        path_file_input = os.path.join(folder_in_kps, file_idx)
        filename = file_idx.split('.')[0]

        with open(path_file_input, 'rb') as f:
            kps_video_npy = np.load(f)
            if kps_video_npy.size == 0 or len(kps_video_npy.shape) != 3:
                print(f"Skipping empty or invalid file: {file_idx}")
                continue
            folder_filepath_out_example = os.path.join(folder_out_features, 'joints_',(type_kps + filename + '.npy'))

            if not os.path.exists(folder_filepath_out_example):
                data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_extended, data_angles_center, data = gen_features.getFeatures(kps_video_npy)
                gen_features.saveFeatures(type_kps, folder_out_features, filename, data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_extended, data_angles_center, data)
            else:
                pass

        
        # try:
        #     data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_center, data = gen_features.getFeatures(kps_video_npy)
        #     gen_features.saveFeatures(type_kps, folder_out_features, filename, data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_center, data)
        # except:
        #     print('path_file_input: ', path_file_input)


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in_kps', required=True, type=str)
    parser.add_argument('--folder_out_features', required=True, type=str)

    parser.add_argument('--type_kps', required=False, default='C4_xyzc', type=str)
    parser.add_argument('--offset', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--noFramesLimit', action='store_true')

    parser.add_argument('--jump_reset', action='store_false')
    

    config_path = 'config.yaml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print(cfg)
    

    args = parser.parse_args()
    print(args)

    main(args, cfg)



""" 

    
"""