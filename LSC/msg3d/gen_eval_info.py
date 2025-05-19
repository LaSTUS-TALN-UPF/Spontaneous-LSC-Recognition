import argparse
import sys
import os

def main(args):
    split = arg.split
    stream = arg.stream
    dataset = arg.dataset
    train_work_dir = arg.train_work_dir
    epoch = arg.epoch
    use_train_norm = arg.use_train_norm
    num_classes = arg.num_classes

    weights = "work_dir/"+train_work_dir+"/weights/weights-"+epoch+".pt"
    if (split=="VAL"):
        work_eval_dir = "eval/"+train_work_dir
        config = "config/TRAIN_CUSTOM/val.yaml"
    elif (split=="TEST"):
        work_eval_dir = "test/"+train_work_dir
        config = "config/TRAIN_CUSTOM/test.yaml"
    elif (split=="TRAIN_SCORE"):
        work_eval_dir = "train_score/"+train_work_dir
        config = "config/TRAIN_CUSTOM/train_score.yaml"
    else:
        print("ERROR SPLIT UNKOWNED")
        sys.exit()

    cmd = "python main_GTM.py --work-dir " + work_eval_dir + " --config " + config + " --weights " + weights + " --device 0 --test-batch-size 250 --seed 42 --stream "+stream+" --dataset "+dataset +" --num-classes "+num_classes
    if use_train_norm:
        cmd += " --use-train-normalization work_dir/"+train_work_dir

    print('cmd: ', cmd)
    os.system(cmd)
    print('END: ', work_eval_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--stream', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--train_work_dir', required=True, type=str)
    parser.add_argument('--epoch', required=True, default='', type=str)
    parser.add_argument('--use_train_norm', action='store_true')
    parser.add_argument('--num_classes', required=True, type=str)

    arg = parser.parse_args()
    # print (arg)
    main(arg)

# python gen_eval_info.py --split VAL --stream joints_C4_xyzc --train_work_dir TRAIN_SIGNAMED/IMAGE_05/joints_C4_xyzc-T1 --epoch 142  
# python gen_eval_info.py --split TEST --stream joints_C4_xyzc --train_work_dir TRAIN_SIGNAMED/IMAGE_05/joints_C4_xyzc-T1 --epoch 142  



# joints_C4_xyzc	MONGAS	TRAIN_SIGNAMED/IMAGE_05/joints_C4_xyzc-T1	0,921
