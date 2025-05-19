from api.genKeypointsPoseHands import GenKeypointsPoseHands
# from api.genKeypointsHolistic import GenKeypointsHolistic
from api.genKeypointsHolisticLegacy import GenKeypointsHolisticLegacy
import argparse
import os
import time
from tqdm import tqdm
import pickle
import cv2
import yaml
import utils

import mediapipe as mp

print("Probando disponibilidad de GPU en MediaPipe...")

try:
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        print("GPU detectada y funcional en MediaPipe")
except Exception as e:
    print(f"Error: {e}")
# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,0,0))


def gen_data_mediapipe(gen_mediapipe, path_input_videos, path_output_data):
    print(path_input_videos)
    
    arr_videos_files = os.listdir(path_input_videos)
    for video_file_idx in tqdm(arr_videos_files):
        video_in_path = os.path.join(path_input_videos, video_file_idx)

        out_path = os.path.join(path_output_data, video_file_idx.replace('.mp4', '.pkl'))
        


        if os.path.exists(out_path):
            continue

        
        cap = cv2.VideoCapture(video_in_path)
        arr_mediapipe_results = []

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            mediapipe_results = {}
            for gen_mediapipe_idx in gen_mediapipe:
                mp_image = gen_mediapipe_idx.load_image(image)
                mediapipe_results.update(gen_mediapipe_idx.get_mediapipeResults(mp_image))

            arr_mediapipe_results.append(mediapipe_results)

        with open(out_path, 'wb') as f:
            pickle.dump(arr_mediapipe_results, f)
        




def main(args, cfg):
    folder_input_videos = args.folder_input_videos
    pose_hands = args.pose_hands
    holistic = args.holistic
    holistic_legacy = args.holistic_legacy

    folder_output_mediapipe = args.folder_output_mediapipe

    print('######################################')
    print('ARGUMENTOS ENTRADA: ')
    print('folder_input_videos: ',folder_input_videos)
    print('pose_hands: ',pose_hands)
    print('holistic: ',holistic)
    print('holistic_legacy: ',holistic_legacy)
    print('folder_output_mediapipe: ',folder_output_mediapipe)
    print('confing_file', config_path)
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
    # if counter_activated > 1: 
    #     raise ValueError("You must specify only one option [ --pose_hands, --holistic, --holistic_legacy ]")

    utils.create_folder(folder_output_mediapipe, reset=False)

    gen_mediapipe = []
    if (pose_hands):
        gen_mediapipe.append(GenKeypointsPoseHands(cfg))
    # if (holistic):
    #     gen_mediapipe.append(GenKeypointsHolistic(cfg))
    if (holistic_legacy):
        gen_mediapipe.append(GenKeypointsHolisticLegacy(cfg))
    
    gen_data_mediapipe(gen_mediapipe, folder_input_videos, folder_output_mediapipe)



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_input_videos', required=True, type=str)
    parser.add_argument('--pose_hands', action='store_true')
    parser.add_argument('--holistic', action='store_true')
    parser.add_argument('--holistic_legacy', action='store_true')
    parser.add_argument('--folder_output_mediapipe', required=True, type=str)
    parser.add_argument('--config_file', required=False, type=str, default='config.yaml')  # NUEVO

    arg = parser.parse_args()

    config_path = arg.config_file
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print(cfg)
    
    main(arg, cfg)


""" 


"""

