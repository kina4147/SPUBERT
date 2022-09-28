import argparse

import os
import numpy as np
from trajnetbaselines.bert2bert.dataset import SocialBERTUCYETHDataset, SocialBERTSTARDataset
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='DATA_BLOCK', help='glob expression for data files')
    parser.add_argument('--output_path', default='OUTPUT_BLOCK', help='glob expression for data files')
    parser.add_argument('--dataset_path', default='ucyeth', help='ucyeth or trajnet')
    parser.add_argument("--dataset_name", default='test', help='output file')
    parser.add_argument('--output', default=None, help='output file')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of oservation')
    parser.add_argument("--obs_length", type=int, default=100, help="number of observation frames")
    parser.add_argument("--pred_length", type=int, default=0, help="number of prediction frames")
    parser.add_argument("--num_neighbor", type=int, default=1, help="number of neighbors")
    parser.add_argument("--num_worker", type=int, default=8, help="dataloader worker size")
    parser.add_argument("--dx", type=int, default=100, help="dataloader worker size")
    parser.add_argument("--dy", type=int, default=100, help="dataloader worker size")
    parser.add_argument("--sx", type=float, default=1.2, help="dataloader worker size")
    parser.add_argument("--sy", type=float, default=1.2, help="dataloader worker size")
    args = parser.parse_args()
    args.dataset_path = os.path.join(args.data_path, args.dataset_path)
    args.nbr_pred = True
    args.spatial_range = 20.0
    args.normalize_scene = False
    if 'ucyeth' in args.dataset_path:
        test_dataset = SocialBERTUCYETHDataset(path=args.dataset_path, name=args.dataset_name, pretrain=False, subset='test', config=args)
        # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    elif 'ethucy_star' in args.dataset_path:
        test_dataset = SocialBERTSTARDataset(path=args.dataset_path, name=args.dataset_name, pretrain=False, subset='test', config=args)
        # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
    scene_img_path = os.path.join(args.dataset_path, args.dataset_name+'.png')
    scene_img = plt.imread(scene_img_path)
    plt.imshow(scene_img)
    all_trajs = []
    if args.dataset_name == 'hotel':
        H = np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                      [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                      [1.1190700e-04, 1.3617400e-05,  5.4276600e-01]])
        args.sx = 1
        args.sy = 1
        args.dx = 0
        args.dy = 0
    elif args.dataset_name == 'eth':
        H = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                      [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                      [3.4555400e-04, 9.2512200e-05,  4.6255300e-01]])
        args.sx = 1
        args.sy = 1
        args.dx = 0
        args.dy = 0
    else:
        H=np.array([[0.02104651, 0., -10.01813922],
                    [0., 0.02386598, -2.79231966],
                    [0., 0., 1.]])
        args.sx = 1
        args.sy = -1
        args.dx = -480
        args.dy = 680

    H_inv = np.linalg.inv(H)

    for idx in range(len(test_dataset)):
        trajs = test_dataset.get_all_trajectories_in_scene(idx, target_coord=False)
        trajs = trajs.transpose((1, 0, 2))
        for t in range(trajs.shape[0]):
            for p in range(trajs.shape[1]):
                if np.isnan(trajs[t, p, 0]):
                    continue
                pos = np.array([[trajs[t, p, 0]], [trajs[t, p, 1]], [1]])
                loc = np.dot(H_inv, pos)
                trajs[t, p, 0] = args.sx*(loc[0]/loc[2]) + args.dx
                trajs[t, p, 1] = args.sy*(loc[1]/loc[2]) + args.dy
        all_trajs.append(trajs)
        if len(all_trajs) > args.num_neighbor:
            break

    # if loc.ndim > 1:
    #     locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
    #     loc_tr = np.transpose(locHomogenous)
    #     loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
    #     locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
    #     return locXYZ[:, :2].astype(int)



    all_trajs = np.concatenate(all_trajs, axis=1)
    if args.dataset_name in ['eth', 'hotel']:
        plt.plot(all_trajs[:, :, 1], all_trajs[:, :, 0], c='black', lw=2.5)
        plt.plot(all_trajs[:, :, 1], all_trajs[:, :, 0], c='yellow', lw=0.5)
    else:
        plt.plot(all_trajs[:, :, 0], all_trajs[:, :, 1], c='black', lw=2.5)
        plt.plot(all_trajs[:, :, 0], all_trajs[:, :, 1], c='yellow', lw=0.5)
    if args.dataset_name == 'eth':
        plt.xlim([0, 640])
        plt.ylim([480, 0])
    else:
        plt.xlim([0, 720])
        plt.ylim([576, 0])
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.dataset_name+'_trajs.png', bbox_inches="tight", pad_inches=0)
    # plt.show()


