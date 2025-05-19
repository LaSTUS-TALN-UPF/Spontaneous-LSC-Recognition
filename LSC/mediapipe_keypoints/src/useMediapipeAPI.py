from api.genKeypointsPoseHands import GenKeypointsPoseHands
# from api.genKeypointsHolistic import GenKeypointsHolistic
from api.genKeypointsHolisticLegacy import GenKeypointsHolisticLegacy
from api.genFeatures import GenFeaturesMediapipeC4, MediapipeOptions
import matplotlib.pyplot as plt
import yaml
import cv2
import numpy as np
import glob
import os
import utils
import argparse


def main(args, cfg):

    path_video_in = args.video_path
    model = args.model
    folder_out_keypoints = args.folder_out_keypoints
    folder_out_video = args.folder_out_video
    folder_out_features = args.folder_out_features
    type_kps = args.type_kps

    # Si hemos pasado una ruta para folder_out_keypoints entonces activamos este flag para obtener y guardar los keypoints.
    GENERATE_KEYPOINTS = (folder_out_keypoints is not None) 
    # Si hemos pasado una ruta para folder_out_video entonces activamos este flag para obtener y guardar frames generados y el vídeo resultante.
    GENERATE_VIDEO = (folder_out_video is not None)
    # Si hemos pasado ruta para la carpeta donde almacenar todos los datos / features generados.
    GENERATE_FEATURES = (folder_out_features is not None)
    

    # Llamamos a la solución de Mediapipe que queremos usar. En función del la clase instanciada utilizaremos uno u otro modelo pero la API será similar.
    if model == 'HOLISTIC':
        gen_mediapipe = GenKeypointsHolistic(cfg)
    elif model == 'POSE_HANDS':
        gen_mediapipe = GenKeypointsPoseHands(cfg)
    elif model == 'HOLISTIC_LEGACY':
        gen_mediapipe = GenKeypointsHolisticLegacy(cfg)

    # PASO1. ABRIMOS EL VIDEO CON OPENCV Y LO RECORREMOS FRAME A FRAME 
    # PASO2. OBTENEMOS LA SALIDA DE MEDIAPIPE 
    # PASO3. OPCIONAL: MOSTRAMOS LA SALIDA DE MEDIAPIPE SOBRE EL FRAME DE ENTRADA Y GENERAMOS VIDEO.
    # PASO4. OPCIONAL: GENERAMOS ARRAY DE KPS UTILIZADO EN SIGNAMED.
    # PASO5. OPCIONAL: VISUALIZAR ARRAY_KPS (2D Y 3D)

    # PASO1. Cargamos el vídeo y lo recorremos llevando un registro del frame actual en caso de queremos guardar los frames con los keypoints.
    kps_video = []
    cap = cv2.VideoCapture(path_video_in)
    filename = os.path.basename(path_video_in).split('.')[0]

    print('Processing ... ', path_video_in)

    if GENERATE_VIDEO:
        print('Generate video from mediapipe output is ACTIVED')
        path_folder_video = os.path.join(folder_out_video, filename)
        utils.create_folder(path_folder_video)
        path_folder_imgs_frames = os.path.join(path_folder_video, 'frames')
        utils.create_folder(path_folder_imgs_frames, reset=False, auto=True)
    else: 
        print('Generate video is DEACTIVED')

    if GENERATE_KEYPOINTS:
        print('Generate keypoints is ACTIVED')
        utils.create_folder(folder_out_keypoints, reset=False)
    else:
        print('Generate keypoints is DEACTIVED')

    if GENERATE_FEATURES:
        print('Generate features is ACTIVED')
        utils.create_folder(folder_out_features, reset=False)
    else:
        print('Generate features is DEACTIVED')


    counter_frame = 0

    while cap.isOpened():
        success, image = cap.read()
        counter_frame += 1
        if not success:
            break

        # PASO2. Adecuamos la imagen para que sea compatible con la versión de Mediapipe que estamos utilizando.
        mp_image = gen_mediapipe.load_image(image)
        
        # PASO2. Obtenemos los keypoints de la imagen.
        mediapipe_results = gen_mediapipe.get_mediapipeResults(mp_image)
        
        # PASO3.1 Mostramos la salida de Mediapipe sobre el frame de entrada y guardamos la imagen en una carpeta temporal.      
        if GENERATE_VIDEO:
            mp_image = gen_mediapipe.draw_landmarks_on_frame(image, mediapipe_results)
            path_out_img_frames_idx = os.path.join(path_folder_imgs_frames, 'frame{}.png'.format(str(counter_frame).zfill(4)))
            cv2.imwrite(path_out_img_frames_idx, mp_image)

        # PASO4.1 Generamos el array de keypoints. 
        data_pose, data_left_hand, data_right_hand = gen_mediapipe.get_results(mediapipe_results, world=False)
        kps_frame = gen_mediapipe.gen_keypoints(data_pose, data_left_hand, data_right_hand, type='SIGNAMED')
        kps_video.append(kps_frame)


    # PASO3.2 Generamos un video con los frames situado en la carpeta temporal.
    if GENERATE_VIDEO:
        width, height, _ = mp_image.shape
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        path_out_video = os.path.join(path_folder_video, filename+'.mp4')
        out = cv2.VideoWriter(path_out_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        
        img_files = sorted(glob.glob('{}/*.png'.format(path_folder_imgs_frames)))
        for img_file in img_files:
            img = cv2.imread(img_file)
            out.write(img)
        out.release()

    if GENERATE_KEYPOINTS:
        # PASO4.2 Guardamos el array de keypoints en un fichero .npy
        kps_video_npy = np.array(kps_video)
        path_out_keypoints = os.path.join(folder_out_keypoints, filename+'.npy')
        np.save(path_out_keypoints, kps_video_npy)

    if GENERATE_FEATURES:
        # PASO4.2 Guardamos el array de keypoints en un fichero .npy
        kps_video_npy = np.array(kps_video)

        if (type_kps=='C3_xyc'):
            gen_features = GenFeaturesMediapipeC4(cfg, MediapipeOptions.XYC)
        elif (type_kps=='C3_xyz'):
            gen_features = GenFeaturesMediapipeC4(cfg, MediapipeOptions.XYZ)
        elif (type_kps=='C4_xyzc'):
            gen_features = GenFeaturesMediapipeC4(cfg, MediapipeOptions.XYZC)

        data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_center, data = gen_features.getFeatures(kps_video_npy)
        gen_features.saveFeatures(type_kps, folder_out_features, filename, data_joints, data_bones, data_motion_joints, data_motion_bones, data_motion_joints5, data_motion_bones5, data_angles, data_angles_center, data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['HOLISTIC', 'POSE_HANDS', 'HOLISTIC_LEGACY'])
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--folder_out_video', required=False, type=str)
    parser.add_argument('--folder_out_keypoints', required=False, type=str)
    parser.add_argument('--folder_out_features', required=False, type=str)
    parser.add_argument('--type_kps', required=False, default='C4_xyzc', type=str)

    args = parser.parse_args()

    # En config.yml se encuentra los datos de configuración propios del uso/trabajo con kps.
    config_path = 'config.yaml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print(cfg)

    main(args, cfg)


""" 

    python useMediapipeAPI.py --model HOLISTIC \
        --video_path /home/tmpvideos/mvazquez/ISLR_workspace/KEYPOINTS_MEDIAPIPE/videos_in/ania_lejos.mp4 \
        --folder_out_video ../out_videos \
        --folder_out_keypoints ../out_keypoints

    python useMediapipeAPI.py --model HOLISTIC_LEGACY \
        --video_path /home/tmpvideos/mvazquez/ISLR_workspace/KEYPOINTS_MEDIAPIPE/videos_in/ania_lejos.mp4 \
        --folder_out_video ../out_videos \
        --folder_out_keypoints ../out_keypoints
    
    python useMediapipeAPI.py --model POSE_HANDS \
        --video_path /home/tmpvideos/mvazquez/ISLR_workspace/KEYPOINTS_MEDIAPIPE/videos_in/ania_lejos.mp4 \
        --folder_out_keypoints ../out_keypoints \
        --folder_out_features ../out_features

 """






    
    




