import random
# import scipy.stats
import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

def flip(data_numpy):
    
    flipped_data = np.copy(data_numpy)
    flipped_data[0,:,:,:] *= -1
    
    return flipped_data


def random_flip(data_numpy):
    
    if (np.random.rand()>0.5):
        return flip(data_numpy)
    
    return data_numpy


# Default dropkps 0%
# def dropkps(data_numpy, dropkps=0):

#     # WIRST LEFT 
#     id_wirst_left = 9
#     id_wirst_right = 10

#     id_hand_left = list(range(19, 40))
#     id_hand_right = list(range(40, 61))


#     data = np.copy(data_numpy)
#     C, T, V, M = data_numpy.shape

#     # Define the keypoints for left and right hand
#     hand_left = list(range(19, 40))
#     hand_right = list(range(40, 61))

#     # Generate random numbers for each frame
#     random_numbers = np.random.rand(T)
#     # Find frames where the random number is less than the threshold
#     dropout_frames = np.where(random_numbers < threshold)[0]
#     # print('DROPOUT_FRAMES: ', dropout_frames)

#     # For each dropout frame, randomly decide whether to dropout left or right hand
#     for frame in dropout_frames:
#         dropout_hand = hand_left if np.random.rand() < 0.5 else hand_right
#         data[:, frame, dropout_hand, :] = 0
#     return data

# dhf: drophand probability fixed
# dhw: drophand probability weighted
def drophand(data_numpy, dhf=0.05, dhw=0.1):

    # ID WIRSTS
    id_wirst_left = 9
    id_wirst_right = 10

    # ID HANDS
    arr_id_hand_left = np.arange(19, 40)
    arr_id_hand_right = np.arange(40, 61)

    data = np.copy(data_numpy)
    C, T, V, M = data_numpy.shape

    # Extract visibility of pose, left hand and right hand - principal keypoint from each part
    print('v_pose.shape: ', data.shape)
    v_pose = (1 - data[-1, :, 0, 0])
    v_hand_left = data[-1, :, 19, 0]
    v_hand_right = data[-1, :, 40, 0]

    print('v_pose: ', v_pose)
    print('v_hand_left: ', v_hand_left)
    print('v_hand_right: ', v_hand_right)

    prob_hand_left = (1 - v_hand_left) * dhw
    prob_hand_right = (1 - v_hand_right) * dhw

    print('prob_hand_left: ', prob_hand_left)
    print('prob_hand_right: ', prob_hand_right)

    arr_presence_hand_left = np.where(v_hand_left > 0)[0]
    arr_presence_hand_right = np.where(v_hand_right > 0)[0]

    print('arr_presence_hand_left: ', arr_presence_hand_left)
    print('len(arr_presence_hand_left): ', len(arr_presence_hand_left))
    print('arr_presence_hand_right: ', arr_presence_hand_right)
    print('len(arr_presence_hand_right): ', len(arr_presence_hand_right))

    # First, filter based on the score
    drop_probability_hand_left = np.random.rand(len(arr_presence_hand_left))
    arr_drop_hand_left_weighted = arr_presence_hand_left[drop_probability_hand_left < prob_hand_left[arr_presence_hand_left]]

    drop_probability_hand_right = np.random.rand(len(arr_presence_hand_right))
    arr_drop_hand_right_weighted = arr_presence_hand_right[drop_probability_hand_right < prob_hand_right[arr_presence_hand_right]]

    # Then, from the remaining ones, select a fixed rate
    remaining_indices_hand_left = np.setdiff1d(arr_presence_hand_left, arr_drop_hand_left_weighted)
    remaining_indices_hand_right = np.setdiff1d(arr_presence_hand_right, arr_drop_hand_right_weighted)

    arr_drop_hand_left_fixed = np.random.choice(remaining_indices_hand_left,  int(dhf * len(remaining_indices_hand_left)), replace=False)
    arr_drop_hand_right_fixed = np.random.choice(remaining_indices_hand_right,  int(dhf * len(remaining_indices_hand_right)) , replace=False)

    # Combine the weighted and fixed drop indices
    arr_drop_hand_left_final = np.concatenate((arr_drop_hand_left_weighted, arr_drop_hand_left_fixed))
    arr_drop_hand_right_final = np.concatenate((arr_drop_hand_right_weighted, arr_drop_hand_right_fixed))

    # print('arr_drop_hand_left_weighted: ', arr_drop_hand_left_weighted)
    # print('len(arr_drop_hand_left_weighted): ', len(arr_drop_hand_left_weighted))
    # print('arr_drop_hand_right_weighted: ', arr_drop_hand_right_weighted)
    # print('len(arr_drop_hand_right_weighted): ', len(arr_drop_hand_right_weighted))

    # print('arr_drop_hand_left_fixed: ', arr_drop_hand_left_fixed)
    # print('len(arr_drop_hand_left_fixed): ', len(arr_drop_hand_left_fixed))
    # print('arr_drop_hand_right_fixed: ', arr_drop_hand_right_fixed)
    # print('len(arr_drop_hand_right_fixed): ', len(arr_drop_hand_right_fixed))

    # print('arr_drop_hand_left_final: ', arr_drop_hand_left_final)
    # print('len(arr_drop_hand_left_final): ', len(arr_drop_hand_left_final))
    # print('arr_drop_hand_right_final: ', arr_drop_hand_right_final)
    # print('len(arr_drop_hand_right_final): ', len(arr_drop_hand_right_final))

    # print('--------------------------------------------------')
    # print('drophand_fixed: ', dhf)
    # print('drophand_weighted: ', dhw)
    # print('--------------------------------------------------')


    # REMOVE KPS HANDS SELECTED - SET WIRST KPS FOR EACH HAND'S KEYPOINTS
    for drop_hand_left_idx in arr_drop_hand_left_final:
        for id_hand_left_idx in arr_id_hand_left:
            data[:, drop_hand_left_idx, id_hand_left_idx, :] = data[:, drop_hand_left_idx, id_wirst_left, :]
            data[3, drop_hand_left_idx, id_hand_left_idx, :] = 0

    for drop_hand_right_idx in arr_drop_hand_right_final:
        for id_hand_right_idx in arr_id_hand_right:
            data[:, drop_hand_right_idx, id_hand_right_idx , :] = data[:, drop_hand_right_idx, id_wirst_right, :]
            data[3, drop_hand_right_idx, id_hand_right_idx, :] = 0

    return data


def resizer(data_numpy, resize):
    data = np.copy(data_numpy)
    data *= resize
    return data

def random_resizer(data_numpy):
    random_size = np.random.uniform(low=0.9, high=1.1)
    return resizer(data_numpy, random_size)

def use_tta(data_numpy, tta):
    
    # print('USE_TTA: ', tta)

    data = np.copy(data_numpy)
    if (tta[0] == True):
        data = flip(data)
    data = resizer(data, tta[1])
    return data

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy

def calculate_init_frame(data_frames, size_max, move_center):
    lower = int(size_max/2)
    upper = int(data_frames - size_max/2)
    
    if (move_center == True):
        mu = ((upper - lower) / 2) + lower
        sigma = (upper - lower) / 20
        center_idx = int(scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=1))
        return center_idx
    else:
        return ((upper - lower) / 2) + lower


def crop_center(data_numpy, size_max, move_center):
    C, T, V, M = data_numpy.shape
    data = np.copy(data_numpy)
    
    if T<size_max:
        for idx in range (size_max - T):
            data = np.insert(data, data.shape[0], 0, axis=0)
    elif T>size_max:
        # idx_min = int(T / 2) - int(size_max/2)
        idx_init = calculate_init_frame(T, size_max, move_center)
        data = data_numpy[:,0:idx_init+size_max,:,:]
        
    return data

    # C, T, V, M = data_numpy.shape


if __name__ == "__main__":

    data_numpy = np.random.rand(4, 50, 61, 1)
    # result = dropkps(data_numpy, 0.5)
    result = drophand(data_numpy, drophand_fixed=0.1, drophand_score=0.5)
    # print(result)