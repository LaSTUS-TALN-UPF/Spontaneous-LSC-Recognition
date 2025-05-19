# setup_paths.sh
#!/bin/bash

# Rutas base
BBDD_PATH="./mediapipe_keypoints/src"
PATH_ANNOTATIONS="$BBDD_PATH/annotations"
PATH_VIDEOS="$BBDD_PATH/VIDEOS"
PATH_MEDIAPIPE="$BBDD_PATH/MEDIAPIPE"
PATH_KEYPOINTS_HL="$BBDD_PATH/KEYPOINTS"
PATH_FEATURES_HL_NORM="$BBDD_PATH/FEATURES"
PATH_DATASET_HL_NORM="$BBDD_PATH/DATASET"

# Variables de entrenamiento
STREAM="joints_C3_xyc"
DATASET='/home/naiara/Documentos/GitHub/SWL-LSE/mediapipe_keypoints/src/data_face'
DEVICE=0
NUM_CLASSES=105
SEED=42
ESTUDIO="face"
CONFIG="config/TRAIN_CUSTOM/train.yaml"
EXPERIMENT="RESULTS/$ESTUDIO/$STREAM-T1"
# \ work_dir/TRAIN_TEST_SWL/E0/42/joints_C3_xyc-T1/weights/weights-111.pt
# Comando de entrenamiento
#     --ignore-weights 'fc.bias' 'fc.weight'\     --weights 'work_dir/TRAIN_TEST/E8_pre_juntos_105/42/joints_C3_xyc-T1/weights/weights-36.pt'\
python main.py \
    --work-dir work_dir/$EXPERIMENT \
    --config $CONFIG \
    --dataset $DATASET \
    --stream $STREAM \
    --num-classes $NUM_CLASSES \
    --device $DEVICE \
    --batch-size 4 \
    --forward-batch-size 4 \
    --test-batch-size 4 \
    --nesterov true \
    --weight-decay 0.0005 \
    --base-lr 0.01 \
    --seed $SEED \
    --use-deterministic \
    --num-worker 8 \
    --early-stopping 30 \
    --step 250 \
    --num-epoch 500 \
    --optimizer 'SGD' \
    --lr-scheduler ReduceLROnPlateau \
    --factor 0.5 \
    --patience 10 \
    --cooldown 0 \



    
    
