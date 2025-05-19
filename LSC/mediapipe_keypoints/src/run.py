import subprocess

# Define the paths
PATH_VIDEOS = "/home/naiara/Descargas/Videos80"
PATH_MEDIAPIPE = "mediapipe"

PATH_KEYPOINTS_HL= 'genenate_arr_hol'
PATH_FEATURES_HL_NORM= 'genenate_arr_hol_norm'
PATH_FEATURES_HL_NO_NORM= 'genenate_arr_hol_no_norm'
PATH_DATASET_HL_NORM= 'dataset_norm'
PATH_DATASET_HL_NO_NORM= 'dataset_no_norm'
PATH_ANNOTATIONS='annotations'

# Step 1: Run generate_mediapipe.py
command1 = [
    "python", "generate_mediapipe.py",
    "--pose_hands", "--holistic_legacy",
    "--folder_input_videos", PATH_VIDEOS,
    "--folder_output_mediapipe", PATH_MEDIAPIPE
]

# Step 2: Run generate_arr_keypoints.py
command2 = [
    "python", "generate_arr_keypoints.py",
    "--holistic_legacy",
    "--folder_input_mediapipe", PATH_MEDIAPIPE,
    "--folder_output_kps", PATH_KEYPOINTS_HL
]

# Step 3: Run generate_features.py
command3 = [
    "python", "generate_features.py",
    "--type_kps", "C3_xyc",
    "--offset", "--normalize",
    "--folder_in_kps", PATH_KEYPOINTS_HL,
    "--folder_out_features", PATH_FEATURES_HL_NORM

]

command4 = [
    "python", "generate_dataset.py",
    "--folder_npy",PATH_FEATURES_HL_NORM,
    "--folder_labels", PATH_ANNOTATIONS,
    "--folder_out", PATH_DATASET_HL_NORM
]

# Run each command one by one
try:
    subprocess.run(command1, check=True)
    subprocess.run(command2, check=True)
    subprocess.run(command3, check=True)
    subprocess.run(command4, check=True)
    print("All steps completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
