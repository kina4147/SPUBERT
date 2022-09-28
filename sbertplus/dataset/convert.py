import numpy as np
import os
import argparse
import pandas as pd
import PIL.Image as pilimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from scipy.interpolate import interp1d
'''
raw convert
'csv pkl txt' mapped to jpeg or png
load pandas
'''
'''
-------------- px_x (v)
|
|
| px_y (u)
'''


def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2].astype(int)

def image2world(traj_px, H):
    pp = np.stack((traj_px[:, 0], traj_px[:, 1], np.ones(len(traj_px))), axis=1)
    PP = np.matmul(H, pp.T).T
    P_normal = PP / np.repeat(PP[:, 2].reshape((-1, 1)), 3, axis=1)
    return P_normal[:, :2]

#
# H = (np.loadtxt(os.path.join(OPENTRAJ_ROOT, "ucyeth/ETH/seq_eth/H.txt")))
# H_inv = np.linalg.inv(H)
# world2image({TRAJ}, H_inv)  # TRAJ: Tx2 numpy array

def convert_gcs(args):
    data_path = os.path.join(args.dataset_path, args.dataset_name)
    traj_path = os.path.join(args.dataset_path, args.dataset_name, "Annotation")
    traj_files = os.listdir(traj_path)
    # traj_files = [file for file in files if file.endswith(".txt")]
    pcount = 0
    raw_data = []
    for traj_file in traj_files:
        traj_filepath = os.path.join(args.dataset_path, args.dataset_name, "Annotation", traj_file)
        with open(traj_filepath, 'r') as f:
            traj_raw_data = f.read().split()
        pid = int(traj_file.replace('.txt', ''))
        # pcount += 1
        # last_frame = -1

        for i in range(len(traj_raw_data) // 3):
            px = float(traj_raw_data[3 * i])
            py = float(traj_raw_data[3 * i + 1])
            frame = int(traj_raw_data[3 * i + 2])

            # there are trajectory files with non-continuous timestamps
            # they need to be counted as different agents
            # if last_frame > 0 and (frame - last_frame) > 20:
            #     pcount += 1
            # last_frame = frame

            # if selected_frames.start <= frame_id < selected_frames.stop:
            raw_data.append([frame, pid, px, py])

    csv_columns = ["frame_id", "ped_id", "px_x", "px_y"]
    raw_data = pd.DataFrame(np.stack(raw_data), columns=csv_columns)
    raw_data.sort_values(by=['frame_id', 'ped_id'], inplace=True)
    # sort or min
    # print(raw_data)
    # raw_data.frame_id = ((raw_data.frame_id) // args.frame_gap)
    # print(raw_data)

    out_filepath = os.path.join(args.dataset_path, args.dataset_name, 'gcs_px.txt')
    raw_data.to_csv(out_filepath, header=None, index=None, sep=' ', mode='w+')
    img_filepath = os.path.join(args.dataset_path, args.dataset_name, 'reference.jpg')
    img = pilimg.open(img_filepath)
    # plt.imshow(img)
    # plt.scatter(raw_data.px_x, raw_data.px_y)
    # plt.show()

    homo_filepath = os.path.join(args.dataset_path, args.dataset_name, 'H.txt')
    H = np.loadtxt(homo_filepath)
    xy = image2world(raw_data[["px_y", "px_x"]].to_numpy(), H)
    xy = xy * 0.8
    raw_data.px_x = xy[:, 0]
    raw_data.px_y = xy[:, 1]
    raw_data.rename(columns={'px_x': 'pos_x', 'px_y': 'pos_y'}, inplace=True)
    out_filepath = os.path.join(args.dataset_path, args.dataset_name, 'gcs.txt')
    raw_data.to_csv(out_filepath, header=None, index=None, sep=' ', mode='w+')

    # plt.scatter(xy[:, 0], xy[:, 1])
    # plt.show()
    # raw_df_groupby = raw_data.groupby("ped_id")
    # trajs = [g for _, g in raw_df_groupby]
    # # tr0_ = trajs[0]
    # # tr1_ = trajs[1]
    #
    # raw_dataset = pd.DataFrame()
    # for ii, tr in enumerate(trajs):
    #     if len(tr) < 2: continue
    #     # interpolate frames (2x up-sampling)
    #     interp_F = np.arange(tr["frame_id"].iloc[0], tr["frame_id"].iloc[-1], 10).astype(int)
    #     interp_X = interp1d(tr["frame_id"], tr["pos_x"], kind='linear')
    #     interp_X_ = interp_X(interp_F)
    #     interp_Y = interp1d(tr["frame_id"], tr["pos_y"], kind='linear')
    #     interp_Y_ = interp_Y(interp_F)
    #     ped_id = tr["ped_id"].iloc[0]
    #     raw_dataset = raw_dataset.append(pd.DataFrame({"frame_id": interp_F,
    #                                                    "agent_id": ped_id,
    #                                                    "pos_x": interp_X_,
    #                                                    "pos_y": interp_Y_}))




def convert_sdd(args):
    data_path = os.path.join(args.dataset_path, args.dataset_name)
    # load the homography values
    with open(os.path.join(data_path, 'scales.yaml'), 'r') as f:
        scales = yaml.load(f, Loader=yaml.FullLoader)
    csv_columns = ["frame", "id", "px_x", "px_y"]
    split_path = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split)
    files = os.listdir(split_path)
    traj_files = [file for file in files if file.endswith(".txt")]
    for traj_file in traj_files:
        traj_filepath = os.path.join(split_path, traj_file)
        raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
        img_filepath = traj_filepath.replace('.txt', '.jpg')
        img = pilimg.open(img_filepath)

        # plt.imshow(img)
        # plt.scatter(raw_data.px_x, raw_data.px_y)
        # plt.show()
        scene_name = traj_file.replace('.txt', '')
        width, height = img.size
        width = int(np.ceil(scales[scene_name] * width * 10))
        height = int(np.ceil(scales[scene_name] * height * 10))


        img_scaled = img.resize((width, height))
        pos_x = scales[scene_name] * raw_data.px_x * 10
        pos_y = scales[scene_name] * raw_data.px_y * 10
        plt.imshow(img_scaled)
        plt.scatter(pos_x, pos_y)
        plt.show()
import copy
def convert_ethucy(args): # px
    args.dataset_split = "students03"
    # if args.dataset_split in ["hotel", "students03", "zara01", "zara02"]:
    #     traj_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, 'obsmat.txt')
    #     csv_columns = ["frame", "id", "pos_x", "pos_z", "pos_y", "vel_x", "vel_z", "vel_y"]
    #     raw_data = pd.read_csv(traj_filepath, sep=r"\s+", header=None, names=csv_columns)
    #     raw_data.drop(['pos_z', 'vel_x', 'vel_z', 'vel_y'], axis=1, inplace=True)
    #     args.frame_gap = 10
    # elif args.dataset_split in ["eth", "zara03", "uni_examples", "students01"]:
    traj_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, 'obsmat.txt')
    csv_columns = ["frame", "id", "class", "ig1", "ig2", "ig3", "ig4", "ig5", "ig6", "ig7", "ig8", "ig9", "ig10", "pos_x", "pos_z", "pos_y", "ig11"]
    raw_data = pd.read_csv(traj_filepath, sep=r"\s+", header=None, names=csv_columns)
    raw_data.drop(['pos_z', "class", "ig1", "ig2", "ig3", "ig4", "ig5", "ig6", "ig7", "ig8", "ig9", "ig10", "ig11"], axis=1, inplace=True)
    args.frame_gap = 1

    offset_x = 0
    offset_y = 0
    if args.dataset_split == "eth":
        pass
        # temp_x = copy.copy(raw_data.pos_x)
        # raw_data.pos_x = -raw_data.pos_y
        # raw_data.pos_y = temp_x
    elif args.dataset_split == "zara03":
        offset_x = -7.5
        offset_y = 5.5
        temp_x = copy.copy(raw_data.pos_x)
        raw_data.pos_x = raw_data.pos_y
        raw_data.pos_y = temp_x
    elif args.dataset_split == "zara01":
        offset_x = -6.5
        offset_y = 5.5
        temp_x = copy.copy(raw_data.pos_x)
        raw_data.pos_x = raw_data.pos_y
        raw_data.pos_y = temp_x
    elif args.dataset_split == "zara02":
        offset_x = -7.5
        offset_y = -10
        temp_x = copy.copy(raw_data.pos_x)
        raw_data.pos_x = raw_data.pos_y
        raw_data.pos_y = temp_x
    elif args.dataset_split == "uni_examples":
        temp_x = copy.copy(raw_data.pos_x)
        raw_data.pos_x = raw_data.pos_y
        raw_data.pos_y = temp_x
    elif args.dataset_split == "students01":
        temp_x = copy.copy(raw_data.pos_x)
        raw_data.pos_x = -raw_data.pos_y
        raw_data.pos_y = temp_x
        offset_x = 6.5
        offset_y = -5.5
    elif args.dataset_split == "students03":
        temp_x = copy.copy(raw_data.pos_x)
        raw_data.pos_x = -raw_data.pos_y
        raw_data.pos_y = temp_x
        offset_x = 7.5
        offset_y = -5.0
    else:
        # offset_x = -7.5
        # offset_y = 5.5
        # temp_x = copy.copy(raw_data.pos_x)
        # raw_data.pos_x = -raw_data.pos_y
        # raw_data.pos_y = temp_x
        print("no offset")

    raw_data.pos_x += offset_x
    raw_data.pos_y += offset_y

    raw_data.frame = ((raw_data.frame - raw_data.frame[0]) // args.frame_gap) * 10
    out_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, args.dataset_split+'.txt')
    raw_data.to_csv(out_filepath, header=None, index=None, sep=' ', mode='w+')
    homo_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, 'H.txt')
    H = np.loadtxt(homo_filepath)
    H_inv = np.linalg.inv(H)
    xy = np.stack((raw_data.pos_x, raw_data.pos_y), axis=-1)
    uv = world2image(xy, H_inv)
    # raw_data.pos_x = uv[:, 0]
    # raw_data.pos_y = uv[:, 1]

    raw_data.pos_x = uv[:, 1]
    raw_data.pos_y = uv[:, 0]
    out_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, args.dataset_split+'_px.txt')
    raw_data.to_csv(out_filepath, header=None, index=None, sep=' ', mode='w+')

    img_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, 'reference.png')
    img = pilimg.open(img_filepath)
    # plt.imshow(img)
    # plt.scatter(raw_data.pos_x, raw_data.pos_y)
    map_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, args.dataset_split+'_map.png')
    map = pilimg.open(map_filepath)

    corner_uv = np.array([[0, 0], [map.height, 0],[0, map.width], [map.height, map.width]])
    corner_xy = image2world(corner_uv, H)
    map = np.array(map)
    occupied_idxs = np.where(map[:, :, 0] != 255)
    occupied_idxs = np.stack(occupied_idxs)
    # print(occupied_idxs)
    occupied_xys = image2world(occupied_idxs.T, H)
    # print(occupied_idxs)
    # print(map[occupied_idxs])

    min_x, max_x = min(corner_xy[:, 0]), max(corner_xy[:, 0])
    min_y, max_y = min(corner_xy[:, 1]), max(corner_xy[:, 1])
    width = max_x - min_x
    height = max_y - min_y
    fig, ax = plt.subplots()
    ax.add_patch(
        patches.Rectangle(
            (min_x, min_y),
            width,
            height,
            edgecolor='blue',
            fill=False
        ))
    cntr_x = (max_x + min_x) / 2.0
    cntr_y = (max_y + min_y) / 2.0

    # map_width = int(np.ceil(max_x - min_x))
    # map_height = int(np.ceil(max_y - min_y))
    # transformed_img = map.transform(
    #     size=(map_width, map_height),
    #     method=pilimg.PERSPECTIVE,
    #     data=H.ravel(),
    # )

    occupied_xys_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, args.dataset_split+'_map_xy.txt')
    np.savetxt(occupied_xys_filepath, occupied_xys, fmt='%.4f')
    boundary_xys_filepath = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split, args.dataset_split+'_map_corner_xy.txt')
    np.savetxt(boundary_xys_filepath, np.array([width, height, cntr_x, cntr_y]), fmt='%.4f')
    # plt.imshow(transformed_img)
    plt.scatter(occupied_xys[:, 0], occupied_xys[:, 1], marker='.', s=1)
    plt.scatter(xy[:, 0], xy[:, 1], marker='.', s=1)
    plt.show()

ethucy_names = {
    'eth': 'eth',
    'hotel': 'hotel',
    'uni_examples': 'uni_examples',
    'students01': 'students01',
    'students03': 'students03',
    'zara01': 'zara01',
    'zara02': 'zara02',
    'zara03': 'zara03'
}
if __name__ == '__main__':
    print("Dataset Converter.")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.obs_len = 8
    args.pred_len = 12
    args.dataset_path = './data'
    args.dataset_name = 'raw/ethucy'
    args.dataset_split = 'zara01'
    args.frame_gap = 10

    # eth = 6
    # other_ethucy = 10
    # sdd = 12
    # gcs = 20
    # convert_ethucy(args)
    convert_ethucy(args)
