import os
import shutil
import numpy as np
import cv2
import mediapipe as mp

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,0,0))


class GenKeypointsHolisticLegacy():
    def __init__(self, cfg, use_model=True):
        self.detector_holistic = None
        self.cfg = cfg
        if use_model:
            self.load_model()

    def load_model(self):
        self.detector_holistic = mp_holistic.Holistic(
            static_image_mode= self.cfg['HOLISTIC_LEGACY_STATIC_IMAGE_MODE'], 
            min_detection_confidence= self.cfg['HOLISTIC_LEGACY_MIN_DETECTION_CONFIDENCE'], 
            min_tracking_confidence= self.cfg['HOLISTIC_LEGACY_MIN_TRACKING_CONFIDENCE'], 
            smooth_landmarks= self.cfg['HOLISTIC_LEGACY_SMOOTH_LANDMARKS'], 
            model_complexity= self.cfg['HOLISITC_LEGACY_MODEL_COMPLEXITY'])
        
    def load_image(self, image):
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_mediapipeResults(self, image):
        results = {}
        results['holistic_legacy'] = self._get_holisticResults(image)
        return results

    def _get_holisticResults(self, image):
        # Error usando directamente la salida de mediapipe al tratar de guardarlo en un pickle o dill
        # _pickle.PicklingError: Can't pickle <class 'mediapipe.python.solution_base.SolutionOutputs'>: attribute lookup SolutionOutputs on mediapipe.python.solution_base failed
        # Poner la salida del modelo en un dicccionario incluyendo cada una de las parte de las que se compone mediapipe.python.solution_base.SolutionOutputs

        detection_result_holistic = self.detector_holistic.process(image)
        out = {
            'pose_landmarks': detection_result_holistic.pose_landmarks,
            'pose_world_landmarks': detection_result_holistic.pose_world_landmarks,
            'face_landmarks': detection_result_holistic.face_landmarks,
            'left_hand_landmarks': detection_result_holistic.left_hand_landmarks,
            'right_hand_landmarks': detection_result_holistic.right_hand_landmarks,
            'segmentation_mask': detection_result_holistic.segmentation_mask,
        }
        return out
    
    def get_results(self, mediapipe_results, world=False):

        data_pose = []
        data_left_hand = []
        data_right_hand = []

        detection_result_holistic = mediapipe_results['holistic_legacy']
        if (world):
            raise NotImplementedError("The field world not avaible in this solution.")
        else:
            if detection_result_holistic['pose_landmarks']:
                data_pose = detection_result_holistic['pose_landmarks']
                data_left_hand = detection_result_holistic['left_hand_landmarks']
                data_right_hand = detection_result_holistic['right_hand_landmarks']

        return data_pose, data_left_hand, data_right_hand

    def get_offset(self, data):
        offset_hand_left = [0, 0, 0, 0, 0]
        offset_hand_right = [0, 0, 0, 0, 0]
        return offset_hand_left, offset_hand_right

    def draw_landmarks_on_frame(self, image, mediapipe_results):
        results = mediapipe_results['holistic_legacy']
        image = cv2.flip(image, 1)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        return image

    def gen_keypoints(self, data_pose, data_left_hand, data_right_hand, type):
        if type == 'HP':
            idx_kps_pose = list(range(33))
            num_total_kps = len(idx_kps_pose) + 21 + 21
            kps_frame = np.zeros((num_total_kps, 4))

            if data_pose:
                for idx, idx_mediapipe in enumerate(idx_kps_pose):
                    kps_frame[idx][0] = data_pose.landmark[idx_mediapipe].x
                    kps_frame[idx][1] = data_pose.landmark[idx_mediapipe].y
                    kps_frame[idx][2] = data_pose.landmark[idx_mediapipe].z
                    kps_frame[idx][3] = data_pose.landmark[idx_mediapipe].visibility
            
            if data_left_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)][0] = data_left_hand.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = data_left_hand.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = data_left_hand.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)][3] = data_left_hand.landmark[idx].visibility

            if data_right_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)+21][0] = data_right_hand.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = data_right_hand.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = data_right_hand.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)+21][3] = data_right_hand.landmark[idx].visibility
            
            return kps_frame

        elif type == 'SIGNAMED':

            idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            num_total_kps = len(idx_kps_pose) + 21 + 21
            kps_frame = np.zeros((num_total_kps, 4))
            if data_pose:
                for idx, idx_mediapipe in enumerate(idx_kps_pose):
                    kps_frame[idx][0] = data_pose.landmark[idx_mediapipe].x
                    kps_frame[idx][1] = data_pose.landmark[idx_mediapipe].y
                    kps_frame[idx][2] = data_pose.landmark[idx_mediapipe].z
                    kps_frame[idx][3] = data_pose.landmark[idx_mediapipe].visibility

            if data_left_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)][0] = data_left_hand.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = data_left_hand.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = data_left_hand.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)][3] = data_left_hand.landmark[idx].visibility

            if data_right_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)+21][0] = data_right_hand.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = data_right_hand.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = data_right_hand.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)+21][3] = data_right_hand.landmark[idx].visibility

            return kps_frame
