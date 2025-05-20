import pickle
import numpy as np
import os, sys
import csv
import argparse
import shutil
from tqdm import tqdm

def create_folder(folder):
    print ('create_folder: {}'.format(folder))    
    if not os.path.exists(folder):
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    else:
        print("Directory " , folder ,  " already exists")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder) 
        print("Directory " , folder ,  " reset")

def load_Split_Labels(split_csv):
    split = []
    labels = []
    with open(split_csv) as csv_file:
        for row in csv.reader(csv_file, delimiter=','): # each row is a list
            split.append(row[0])
            labels.append(np.int64(row[1]))
    return split, labels


def generate_dataset_subset(path_file_label, path_folder_data, include_data = True, suffix=""):
    split_data = []
    split_ok = []
    labels_ok = []

    print('path_file_label: ', path_file_label)
    print('path_folder_data: ', path_folder_data)

    split, labels = load_Split_Labels(path_file_label)

    for idx in tqdm(range(len(split))):
        try:
            if (include_data):
                file = split[idx]
                path_data = os.path.join(path_folder_data, file + suffix +'.npy')
               
                try:
                    data_idx = np.load(path_data)
                    print(f"Data shape: {data_idx.shape}")  # Debug print
                    split_data.append(data_idx)
                except Exception as e:
                    print(f'Error loading {file}: {str(e)}')
                    continue

            split_ok.append(split)
            labels_ok.append(labels[idx])
        except:
            print ('discard: {}'.format(file))

    np_split_data = np.array(split_data)
    print ('Dataset {} - {} - shape : {}'.format(path_file_label, path_folder_data, len(np_split_data)))
    print ('Discarded labels: {}/{}'.format(len(split)-len(split_ok), len(split)))

    return np_split_data, split_ok, labels_ok

def generate_dataset_subset_memmap(path_file_label, path_folder_data, path_output_data, path_output_labels, dtype=np.float32, suffix=""):
    split, labels = load_Split_Labels(path_file_label)

    # Cargar un archivo para conocer la forma
    first_sample_path = os.path.join(path_folder_data, split[0] + suffix + ".npy")
    first_sample = np.load(first_sample_path)
    shape_sample = first_sample.shape
    total_samples = len(split)

    # Crear memmap para datos y etiquetas
    memmap_data = np.memmap(path_output_data, dtype=dtype, mode='w+', shape=(total_samples, *shape_sample))
    memmap_labels = np.memmap(path_output_labels, dtype=np.int64, mode='w+', shape=(total_samples,))

    print(f"Writing memmap dataset to {path_output_data} with shape {memmap_data.shape}")

    successful = 0
    for i in tqdm(range(total_samples)):
        file = split[i]
        path = os.path.join(path_folder_data, file + suffix + ".npy")
        try:
            sample = np.load(path)
            memmap_data[i] = sample
            memmap_labels[i] = labels[i]
            successful += 1
        except Exception as e:
            print(f"Error with {file}: {e}")

    memmap_data.flush()
    memmap_labels.flush()
    print(f"Final dataset shape: {memmap_data.shape}, loaded: {successful}/{total_samples}")

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            parts = filename.split('_')
            if "test" in parts:
                index = parts.index("test")
            elif "val" in parts:
                index = parts.index("val")
            elif "train" in parts:
                index = parts.index("train")
            else:
                continue

            # Construct new filename with "test_", "val_", or "train_" at the beginning
            new_name = f"{parts[index]}_" + "_".join(parts[:index] + parts[index+1:])
            
            # Remove "_data" if it is at the end
            if new_name.endswith("_data.npy"):
                new_name = new_name.replace("_data.npy", ".npy")
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} to {new_path}")



def main(args, folder_data, splits_data):
    folder_npy = args.folder_npy
    folder_labels = args.folder_labels
    folder_out = args.folder_out
    suffix = args.suffix


    create_folder(folder_out)
    train_split_csv = os.path.join(folder_labels, 'train_labels.csv')
    val_split_csv = os.path.join(folder_labels, 'val_labels.csv')
    test_split_csv = os.path.join(folder_labels, 'test_labels.csv')

    try:
        train_split, train_labels = load_Split_Labels(train_split_csv)
    except:
        print('Problem load train split csv')
    try:
        val_split, val_labels = load_Split_Labels(val_split_csv)
    except:
        print('Problem load val_split_csv')
    try:
        test_split, test_labels = load_Split_Labels(test_split_csv)
    except:
        print('Problem load test_split_csv')

    
    print('#########################################################################################################')
    print('#######################################  GEN SPLIT DATA #################################################')
    print('#########################################################################################################')

    for folder_data_idx in folders_data:
        for split_prefix in splits_data.keys():
            path_file_label = os.path.join(folder_labels, splits_data[split_prefix])
            path_folder_data = os.path.join(folder_npy, folder_data_idx)
            path_split_out = os.path.join(folder_out, (folder_data_idx + '_' + split_prefix + '.npy'))

            np_split_data, split_ok, labels_ok = generate_dataset_subset(path_file_label, path_folder_data, suffix=suffix)
            np.save(path_split_out, np_split_data)
           
            print('>> GENERATED: ', path_split_out)
            print('----------------------------------------------------')


    print('#########################################################################################################')
    print('######################################  GEN SPLIT ANNOTATION ############################################')
    print('#########################################################################################################')

    folder_data_idx = folders_data[0]
    for split_prefix in splits_data.keys():
        path_file_label = os.path.join(folder_labels, splits_data[split_prefix])
        path_folder_data = os.path.join(folder_npy, folder_data_idx)

        split_prefix.replace('data', '')
        path_split_out = os.path.join(folder_out, (split_prefix + folder_data_idx + '.npy'))

        np_split_data, split_ok, labels_ok = generate_dataset_subset(path_file_label, path_folder_data, include_data=False, suffix=suffix)


        filename_annotation = split_prefix.replace('data','label.pkl')
        path_annotation_file_out = os.path.join(folder_out, filename_annotation)
        with open(path_annotation_file_out, 'wb') as f:
            pickle.dump((split_ok, labels_ok), f)

        print('>> GENERATED: ', path_annotation_file_out)
        print('----------------------------------------------------')

    rename_files(folder_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_npy', required=True, type=str)
    parser.add_argument('--folder_labels', required=True, type=str)
    parser.add_argument('--folder_out', required=True, type=str)
    parser.add_argument('--rename', action='store_false')
    parser.add_argument('--suffix', required=False, type=str, default="")

    folders_data = [
        #"bones_C4_xyzc",
        #"bones_motion_C4_xyzc",
        #"joints_motion_C4_xyzc",
        # "bones_motion5_C4_xyzc",
        # "joints_motion5_C4_xyzc",
        #"bones_C4_xyzca",
        #"joints_C4_xyzca",
        # "bones_C4_xyzcg",
        # "joints_C4_xyzcg",
        # "joints_bones_C4_xyzc",
        # "joints_bones_C4_xyzca",
        # "joints_bones_C4_xyzcg",
        # "joints_bones_motion_C4_xyzca",
        # "joints_bones_motion5_C4_xyzca",
        # "joints_bones_motion_C4_xyzcg",
        # "bones_C3_xyc",
        "joints_C3_xyc",
        #"bones_motion_C3_xyc",
        #"joints_motion_C3_xyc",
        # "bones_motion5_C3_xyc",
        # "joints_motion5_C3_xyc",
        #"bones_C3_xyca",
        #"joints_C3_xyca",
        # "bones_C3_xycg",
        # "joints_C3_xycg",
        # "joints_bones_C3_xyc",
        # "joints_bones_C3_xyca",
        # "joints_bones_C3_xycg",
        # "joints_bones_motion_C3_xyca",
        # "joints_bones_motion5_C3_xyca",
        # "joints_bones_motion_C3_xycg",
        # "joints_bones_motion5_C3_xycg"
        #"angles_extended_C3_xyc",
        #"angles_extended_C4_xyzc",
        #"angles_C3_xyc",
        #"angles_C4_xyzc",
        #"angles_extended_C3_xyc",
        #"angles_extended_C4_xyzc",
        ]


    splits_data = {
        'train_data': 'train_labels.csv',
        'val_data': 'val_labels.csv',
        'test_data': 'test_labels.csv'
    }


    arg = parser.parse_args()
    main(arg, folders_data, splits_data)
    

    """

    """
