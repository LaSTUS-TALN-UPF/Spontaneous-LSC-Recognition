import os
import shutil
import numpy as np
from datetime import datetime
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import json
import time


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class GenKeypointsPoseHands():
    def __init__(self, cfg, use_model=True):
        self.detector_pose = None
        self.detector_hands = None
        self.cfg = cfg
        if use_model:
            self.load_model()

    def load_model(self):

        base_options = python.BaseOptions(
            model_asset_path='../models/pose_landmarker_heavy.task',
            delegate = self.cfg['MODEL_DELEGATE']
        )

        options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.VIDEO,
                num_poses=self.cfg['POSE_NUM_POSES'],
                min_pose_detection_confidence=self.cfg['POSE_MIN_POSE_DETECTION_CONFIDENCE'],
                min_pose_presence_confidence=self.cfg['POSE_MIN_POSE_PRESENCE_CONFIDENCE'],
                min_tracking_confidence=self.cfg['POSE_MIN_TRACKING_CONFIDENCE'],
                output_segmentation_masks=self.cfg['POSE_OUTPUT_SEGMENTATION_MASK'],
        )
        self.detector_pose = vision.PoseLandmarker.create_from_options(options)

        base_options = python.BaseOptions(
            model_asset_path='../models/hand_landmarker.task',
            delegate = self.cfg['MODEL_DELEGATE']
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.cfg['HANDS_NUM_HANDS'],
            min_hand_detection_confidence=self.cfg['HANDS_MIN_HAND_DETECTION_CONFIDENCE'],
            min_hand_presence_confidence=self.cfg['HANDS_MIN_HAND_PRESENCE_CONFIDENCE'],
        )
        self.detector_hands = vision.HandLandmarker.create_from_options(options)

    def load_image(self, image):
        image = cv2.flip(image, 1)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def get_mediapipeResults(self, image):
        results = {}
        results['pose'] = self._get_poseResults(image)
        results['hands'] = self._get_handsResults(image)
        return results

    def _get_poseResults(self, image):
        timestamp = int(time.time() * 1000)
        return self.detector_pose.detect_for_video(image, timestamp)

    def _get_handsResults(self, image):
        timestamp = int(time.time() * 1000)
        return self.detector_hands.detect_for_video(image, timestamp)

    def get_results(self, mediapipe_results, world=False):
        detection_result_pose = mediapipe_results['pose']
        detection_result_hands = mediapipe_results['hands']
        data_pose = []
        data_left_hand = []
        data_right_hand = []

        if world:
            data_pose = detection_result_pose.pose_world_landmarks
            hand_landmarks_list = detection_result_hands.hand_world_landmarks
        else:
            data_pose = detection_result_pose.pose_landmarks
            hand_landmarks_list = detection_result_hands.hand_landmarks

        handedness_list = detection_result_hands.handedness
        for idx in range(len(handedness_list)):
            if(handedness_list[idx][0].category_name == 'Left'):
                data_left_hand_idx = hand_landmarks_list[idx]
                for data in data_left_hand_idx:
                    data.visibility = handedness_list[idx][0].score
                data_left_hand.append(data_left_hand_idx)
            if(handedness_list[idx][0].category_name == 'Right'):
                data_right_hand_idx = hand_landmarks_list[idx]
                for data in data_right_hand_idx:
                    data.visibility = handedness_list[idx][0].score
                data_right_hand.append(data_right_hand_idx)

        return data_pose, data_left_hand, data_right_hand
    
    def insert_pose_info(self, kps_frame_idx, data, idx):
        kps_frame_idx[0] = data[idx].x
        kps_frame_idx[1] = data[idx].y
        kps_frame_idx[2] = data[idx].z
        kps_frame_idx[3] = data[idx].visibility
    
    def insert_hands_info(self, kps_frame_idx, data, idx):
        kps_frame_idx[0] = data[idx].x
        kps_frame_idx[1] = data[idx].y
        kps_frame_idx[2] = data[idx].z
        kps_frame_idx[3] = data[idx].visibility

    def get_offset(self, data):
        offset_hand_left = [data[9][0] - data[19][0], data[9][1] - data[19][1], data[9][2] - data[19][2], 0]
        offset_hand_right = [data[10][0] - data[40][0], data[10][1] - data[40][1], data[10][2] - data[40][2], 0]
        return offset_hand_left, offset_hand_right

    def draw_landmarks_on_frame(self, image, mediapipe_results):

        detection_result_pose = mediapipe_results['pose']
        detection_result_hands = mediapipe_results['hands']

        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = self._draw_landmarks_on_frame_pose(cv2.flip(image, 1), detection_result_pose)
        annotated_image = self._draw_landmarks_on_frame_hands(annotated_image, detection_result_hands)
        return annotated_image

    def gen_keypoints(self, data_pose, data_left_hand, data_right_hand, type):
        if type == 'HP':
            idx_kps_pose = list(range(33))
            num_total_kps = len(idx_kps_pose) + 21 + 21
            kps_frame = np.zeros((num_total_kps, 4))

            if data_pose:
                for idx, idx_mediapipe in enumerate(idx_kps_pose):
                    kps_frame[idx][0] = data_pose[0][idx_mediapipe].x
                    kps_frame[idx][1] = data_pose[0][idx_mediapipe].y
                    kps_frame[idx][2] = data_pose[0][idx_mediapipe].z
                    kps_frame[idx][3] = data_pose[0][idx_mediapipe].visibility
            
            if data_left_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)][0] = data_left_hand[0][idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = data_left_hand[0][idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = data_left_hand[0][idx].z
                    kps_frame[idx+len(idx_kps_pose)][3] = data_left_hand[0][idx].visibility

            if data_right_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)+21][0] = data_right_hand[0][idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = data_right_hand[0][idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = data_right_hand[0][idx].z
                    kps_frame[idx+len(idx_kps_pose)+21][3] = data_right_hand[0][idx].visibility
            return kps_frame

        elif type == 'SIGNAMED':

            idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            num_total_kps = len(idx_kps_pose) + 21 + 21
            kps_frame = np.zeros((num_total_kps, 4))

            if data_pose:
                for idx, idx_mediapipe in enumerate(idx_kps_pose):
                    kps_frame[idx][0] = data_pose[0][idx_mediapipe].x
                    kps_frame[idx][1] = data_pose[0][idx_mediapipe].y
                    kps_frame[idx][2] = data_pose[0][idx_mediapipe].z
                    kps_frame[idx][3] = data_pose[0][idx_mediapipe].visibility

            if data_left_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)][0] = data_left_hand[0][idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = data_left_hand[0][idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = data_left_hand[0][idx].z
                    kps_frame[idx+len(idx_kps_pose)][3] = data_left_hand[0][idx].visibility

            if data_right_hand:
                for idx in range(21):
                    kps_frame[idx+len(idx_kps_pose)+21][0] = data_right_hand[0][idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = data_right_hand[0][idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = data_right_hand[0][idx].z
                    kps_frame[idx+len(idx_kps_pose)+21][3] = data_right_hand[0][idx].visibility
            return kps_frame


    def _draw_landmarks_on_frame_pose(self, image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(image)
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
                )
        return annotated_image

    def _draw_landmarks_on_frame_hands(self, rgb_image, detection_result):
        MARGIN = 0  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 3
        HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
                )
            
            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name[0]}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                        
        return annotated_image
        