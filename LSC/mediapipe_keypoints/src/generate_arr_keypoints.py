from api.genKeypointsPoseHands import GenKeypointsPoseHands
# from api.genKeypointsHolistic import GenKeypointsHolistic
from api.genKeypointsHolisticLegacy import GenKeypointsHolisticLegacy
import argparse
import os
import time
from tqdm import tqdm
import cv2
import yaml
import numpy as np
import shutil
import utils
import pickle
# import dill

def gen_data_keypoints(gen_mediapipe, path_folder_input, path_output_data, use_world):
    arr_input_files = os.listdir(path_folder_input)
    for file_idx in tqdm(arr_input_files):
        path_file_mediapipe_results = os.path.join(path_folder_input, file_idx)

        filename = file_idx.replace('.pkl', '.npy')
        path_out_keypoints = os.path.join(path_output_data, filename)
        if os.path.exists(path_out_keypoints):
            continue
        try:
            # Intenta abrir el archivo y procesarlo
            with open(path_file_mediapipe_results, 'rb') as f:
                arr_mediapipe_results = pickle.load(f)

            arr_kps_video = []
            for mediapipe_results in arr_mediapipe_results:
                data_pose, data_left_hand, data_right_hand ,data_face= gen_mediapipe.get_results(mediapipe_results, world=use_world)
                kps_frame = gen_mediapipe.gen_keypoints(data_pose, data_left_hand, data_right_hand,data_face, type='HP')
                arr_kps_video.append(kps_frame)

            arr_kps_video_npy = np.array(arr_kps_video)
            np.save(path_out_keypoints, arr_kps_video_npy)
        
        except Exception as e:
            # Si ocurre un error, muestra el error y continúa con el siguiente archivo
            print(f"Error procesando el archivo {file_idx}: {e}")
           
            continue  # Continúa con el siguiente archivo
       
        # dict_keys(['pose', 'hands', 'holistic_legacy'])



def main(args, cfg):
    pose_hands = args.pose_hands
    holistic = args.holistic
    holistic_legacy = args.holistic_legacy

    folder_input_mediapipe = args.folder_input_mediapipe
    folder_output_kps = args.folder_output_kps

    use_world = args.world
    
    print('######################################')
    print('ARGUMENTOS ENTRADA: ')
    print('pose_hands: ', pose_hands)
    print('holistic: ', holistic)
    print('holistic_legacy: ', holistic_legacy)

    print('folder_input_mediapipe: ', folder_input_mediapipe)
    print('folder_output_kps: ', folder_output_kps)

    print('use_world: ', use_world)
    print('######################################')

    counter_activated = 0
    if pose_hands:
        counter_activated = counter_activated + 1
    if holistic:
        counter_activated = counter_activated + 1
    if holistic_legacy:
        counter_activated = counter_activated + 1

    if counter_activated == 0: 
        raise ValueError("You must specify at least one option [ --pose_hands, --holistic, --holistic_legacy ]")
    if counter_activated > 1: 
        raise ValueError("You must specify only one option [ --pose_hands, --holistic, --holistic_legacy ]")

    utils.create_folder(folder_output_kps)

    if pose_hands:
        genMediapipe = GenKeypointsPoseHands(cfg, use_model=False)
    # if holistic:
    #     genMediapipe = GenKeypointsHolistic(cfg, use_model=False)
    if holistic_legacy:
    
        genMediapipe = GenKeypointsHolisticLegacy(cfg, use_model=False)
    
        
    gen_data_keypoints(genMediapipe, folder_input_mediapipe, folder_output_kps, use_world)
   

        
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_hands', action='store_true')
    parser.add_argument('--holistic', action='store_true')
    parser.add_argument('--holistic_legacy', action='store_true')

    parser.add_argument('--folder_input_mediapipe', default='', type=str)
    parser.add_argument('--folder_output_kps', required=True, type=str)

    parser.add_argument('--world', action='store_true')

    config_path = 'config_face.yaml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print(cfg)
    

    args = parser.parse_args()
    main(args, cfg)


""" 

    
"""
