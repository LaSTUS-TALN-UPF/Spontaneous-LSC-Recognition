# setup_paths.sh
#!/bin/bash


STREAM="joints_C3_xyc"
DATASET='/home/naiara/Documentos/GitHub/SWL-LSE/mediapipe_keypoints/src/data' #dataset_path
DEVICE=0
NUM_CLASSES=105
SEED=42
ESTUDIO="resulta_train" ##Change the name of the experiment
CONFIG="config/TRAIN_CUSTOM/train.yaml"
EXPERIMENT="$ESTUDIO"
WEIGHTS='weights/weights_SWL.pt' # weights of the previous model


python main.py \
    --work-dir $EXPERIMENT \
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
    --weights $WEIGHTS \
    --ignore-weights 'fc.bias' 'fc.weight' \
    --factor 0.5 \
    --patience 10 \
    --cooldown 0 \



    
    