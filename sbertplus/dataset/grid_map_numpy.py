"""

Grid map library using numpy

author: Ki-In Na

"""

import matplotlib.pyplot as plt
import numpy as np
import math

# 전체 그리드를 각도에 따라서 다 갖고 있고, 이들이 거리가, 해당 점의 각도로 각도 인덱스를 뽑아내고, 거리가 가까운 것을 식별한다.
# 각도 인덱스에 그리드맵의 모든 셀을 매핑시킨다.

def filter_valid_index(xy_pos, xy_ids):
    xy_pos = np.delete(xy_pos, xy_ids < 0)
    xy_ids = np.delete(xy_ids, xy_ids < 0)
    return xy_pos, xy_ids

def estimate_num_patch(map_length, patch_size):
    if map_length % patch_size == 0:
        return (map_length // patch_size)**2
    else:
        return ((map_length // patch_size)+2)**2


def estimate_map_length(map_range, map_resol):
    num_idx = map_range / map_resol
    num_idx = int(num_idx) if (num_idx % 2) == 0 else int(num_idx + 1)
    return num_idx

def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle

# def extract_cgrid_map(src_gmap, cntr_x, cntr_y, range_x, range_y, resol):
#     dst_cmap = CGridMap()
#     for (x, y) in zip(ox, oy):
#         d = math.hypot(x, y)
#         # angle = atan_zero_to_twopi(py, px)
#         # angle_id = dst_cmap.get_id_from_angle(angle_id)
#         d_id, a_id = dst_cmap.get_da_idx_from_xy_pos(x, y)

def extract_grid_map(src_gmap, cntr_x, cntr_y, range_x, range_y, resol):
    """
    :param width: number of grid for width
    :param height: number of grid for height
    :param resolution: grid resolution [m]
    :param center_x: center x position  [m]
    :param center_y: center y position [m]
    :param init_val: initial value for all grid
    """

    num_rows = int(2.0 * range_y / resol)
    num_cols = int(2.0 * range_x / resol)
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    dst_gmap = RectangularGridMap(num_cols, num_rows, resol, cntr_x, cntr_y)
    xs, ys = dst_gmap.get_all_positions()
    src_vals, valid = src_gmap.get_value_from_xy_pos(xs, ys)
    xs = xs[valid]
    ys = ys[valid]
    dst_gmap.set_values_from_xy_pos(xs, ys, src_vals)

    return dst_gmap

def test_extract(grid_map):
    # pass
    # GridMap(100, 100, 0.5, center_x=0, center_y=50)
    e_gmap = extract_grid_map(grid_map, 0, 0, 50, 50, 0.5)
    e_gmap.flip_x()
    e_gmap.plot_grid_map_in_space()

    return e_gmap

def test_cgrid_map(grid_map):
    d_range = 100
    d_resol = 1.0
    a_range = np.pi/2
    a_resol = np.pi/90

    num_cols = int(d_range / d_resol)
    num_rows = int(2.0 * a_range / a_resol)
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    cgmap = CircularGridMap(num_cols, num_rows, d_resol, a_resol, center_x=0, center_y=0, center_d=0, center_a=0)
    xs, ys = cgmap.get_all_xy_pos()
    plt.scatter(xs, ys, s=3.0, marker='.')
    for x, y in zip(xs, ys):
        val = grid_map.get_value_from_xy_pos(x, y)
        if not val: continue
        cgmap.set_value_from_xy_pos(x, y, val)
        # cgmap.set_value_from_xy_pos(x, y, 1.0)
    # cgmap.update_ray_casting()
    cgmap.plot_grid_map_in_space()


class RectangularGridMap:
    """
    GridMap class
    """

    def __init__(self, width, height, resolution, center_x, center_y, init_val=0.0):
        """__init__

        :param width: number of grid for width (num_col, num_xs)
        :param height: number of grid for height (num_row, num_ys)
        :param resolution: grid resolution [m]
        :param center_x: center x position  [m]
        :param center_y: center y position [m]
        :param init_val: initial value for all grid
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y
        self.init_val = init_val
        self.none_val = -1
        self.none_idx = -1
        self.none_pos = -1

        # self.min_grid_x = self.center_x - self.width / 2.0 * self.resolution
        # self.min_grid_y = self.center_x - self.width / 2.0 * self.resolution

        self.min_x = self.center_x - self.width / 2.0 * self.resolution
        self.min_y = self.center_y - self.height / 2.0 * self.resolution
        self.max_x = self.center_x + self.width / 2.0 * self.resolution
        self.max_y = self.center_y + self.height / 2.0 * self.resolution

        self.num_grid = self.width * self.height
        # self.data = np.full(self.num_grid, init_val)
        self.grid_map = np.full((self.height, self.width), init_val)
        # self.data = [init_val] * self.ndata

    def get_all_positions(self):
        x_ids, y_ids = self.get_all_indices()
        xs = self.calc_grid_center_position_from_index(x_ids, self.min_x)
        ys = self.calc_grid_center_position_from_index(y_ids, self.min_y)
        return xs, ys

    def get_all_indices(self):
        x_ids, y_ids = np.mgrid[slice(0, self.width, 1),
                                slice(0, self.height, 1)]
        return x_ids.flatten(), y_ids.flatten()

    def calc_xy_index_from_grid_index(self, grid_ids):
        pass

    def calc_grid_index_from_xy_index(self, x_ids, y_ids):
        grid_ids = (y_ids * self.width + x_ids).astype(np.int64)
        return grid_ids

    def calc_grid_center_xy_position_from_xy_index(self, x_ids, y_ids):
        x_pos = self.calc_grid_center_position_from_index(x_ids, self.min_x)
        y_pos = self.calc_grid_center_position_from_index(y_ids, self.min_y)
        return x_pos, y_pos

    def calc_grid_center_position_from_index(self, idx, min_pos):
        return min_pos + idx * self.resolution + self.resolution / 2.0

    def calc_index_from_position(self, pos, min_pos, max_idx):
        ids = (np.floor((pos - min_pos) / self.resolution)).astype(np.int64)
        valid = (0 <= ids) & (ids < max_idx)
        return ids, valid

    def get_value_from_xy_index(self, x_ids, y_ids):
        """get_value_from_xy_index
        when the index is out of grid map area, return None
        :param x_ids: x index
        :param y_ids: y index
        """
        # grid_ids = self.calc_grid_index_from_xy_index(x_ids, y_ids)
        # grid_ids[(grid_ids < 0) | (grid_ids > self.num_grid)] = self.none_idx
        # print(type(y_ids), type(x_ids))
        # print(y_ids.shape, x_ids.shape)

        vals = self.grid_map[y_ids, x_ids]
        return vals

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """

        x_ids, x_valid = self.calc_index_from_position(x_pos, self.min_x, self.width)
        y_ids, y_valid = self.calc_index_from_position(y_pos, self.min_y, self.height)
        valid = x_valid & y_valid
        return x_ids, y_ids, valid

    def update_val_from_xy_pos(self, x_pos, y_pos, up_val):

        """update_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param up_val: update grid value
        """

        x_ids, y_ids, valid = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        # if (not x_ind) or (not y_ind):
        #     return False  # NG
        # out_ids = (x_ids < 0) | (y_ids < 0)
        # np.delete(x_ids)
        # grid_ids = self.calc_grid_index_from_xy_index(x_ids, y_ids)
        # np.delete(grid_ids, (grid_ids < 0) | (grid_ids > self.num_grid))
        # self.data[grid_ids] += up_val
        for x_idx, y_idx in zip(x_ids, y_ids):
            self.grid_map[y_idx, x_idx] += up_val

    def get_value_from_xy_pos(self, x_pos, y_pos):

        x_ids, y_ids, valid = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        x_ids, y_ids = x_ids[valid], y_ids[valid]
        return self.get_value_from_xy_index(x_ids, y_ids), valid


    def set_values_from_xy_pos(self, x_pos, y_pos, vals):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ids, y_ids, valid = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        x_ids, y_ids, vals = x_ids[valid], y_ids[valid], vals[valid]
        self.set_value_from_xy_index(x_ids, y_ids, vals)

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ids, y_ids, valid = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        x_ids, y_ids = x_ids[valid], y_ids[valid]
        self.set_value_from_xy_index(x_ids, y_ids, val)


    def set_value_from_xy_index(self, x_ids, y_ids, vals):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ids: x index
        :param y_ids: y index
        :param val: grid value
        """
        # # print(type(x_ids), type(y_ids), type(vals))
        # print(self.width, self.height, x_ids, y_ids)
        self.grid_map[y_ids, x_ids] = vals

    def flip_x(self):
        # grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        self.grid_map = np.flip(self.grid_map, axis=0)
        # self.data = grid_data.flatten().tolist()

    def flip_y(self):
        # grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        self.grid_map = np.flip(self.grid_map, axis=1)
        # self.data = grid_data.flatten().tolist()

    def rotate(self, theta):
        # xs, ys = self.get_all_positions()
        # vals, valid = self.get_value_from_xy_pos(xs, ys)
        # self.init()
        # xys = np.stack((xs, ys), axis=-1)
        # ct = math.cos(theta)
        # st = math.sin(theta)
        # r = np.array([[ct, st], [-st, ct]])
        # rxys = xys.dot(r.transpose())
        # self.set_values_from_xy_pos(rxys[:, 0], rxys[:, 1], vals)
        xs, ys = self.get_all_positions()
        xys = np.stack((xs, ys), axis=-1)
        ct = math.cos(-theta)
        st = math.sin(-theta)
        r = np.array([[ct, st], [-st, ct]])
        rxys = xys.dot(r.transpose())
        vals, valid = self.get_value_from_xy_pos(rxys[:, 0], rxys[:, 1])
        xs, ys = xs[valid], ys[valid]
        self.init()
        self.set_values_from_xy_pos(xs, ys, vals)

    def init(self):
        # self.grid_map[:, :] = self.init_val
        self.grid_map = np.full((self.height, self.width), self.init_val)

    def check_occupancy_from_xy_pos(self, x_pos, y_pos):
        x_ids, y_ids, valid = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        x_ids, y_ids = x_ids[valid], y_ids[valid]
        vals = self.get_value_from_xy_index(x_ids, y_ids)
        return vals != self.init_val

    def check_occupancy_from_xy_index(self, x_ids, y_ids):
        vals = self.get_value_from_xy_index(x_ids, y_ids)
        return vals != self.init_val

    def set_value_from_polygon(self, pol_x, pol_y, val, inside=True):
        """set_value_from_polygon

        Setting value inside or outside polygon

        :param pol_x: x position list for a polygon
        :param pol_y: y position list for a polygon
        :param val: grid value
        :param inside: setting data inside or outside
        """

        # making ring polygon
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            pol_x.append(pol_x[0])
            pol_y.append(pol_y[0])

        # setting value for all grid



        # for x_ind in range(self.width):
        #     for y_ind in range(self.height):

        x_ids, y_ids = self.get_all_indices()
        x_pos, y_pos = self.calc_grid_center_xy_position_from_xy_index(x_ids, y_ids)
        for x_idx, y_idx, x, y in zip(x_ids, y_ids, x_pos, y_pos):
            flag = self.check_inside_polygon(x, y, pol_x, pol_y)


                # print(x_ind, y_ind, self.width, self.height)
            if flag is inside:
                self.set_value_from_xy_index(x_idx, y_idx, val)

    # def expand_grid(self):
    #     xinds, yinds = [], []
    #
    #     for ix in range(self.width):
    #         for iy in range(self.height):
    #             if self.check_occupied_from_xy_index(ix, iy):
    #                 xinds.append(ix)
    #                 yinds.append(iy)
    #
    #     for (ix, iy) in zip(xinds, yinds):
    #         self.set_value_from_xy_index(ix + 1, iy, val=1.0)
    #         self.set_value_from_xy_index(ix, iy + 1, val=1.0)
    #         self.set_value_from_xy_index(ix + 1, iy + 1, val=1.0)
    #         self.set_value_from_xy_index(ix - 1, iy, val=1.0)
    #         self.set_value_from_xy_index(ix, iy - 1, val=1.0)
    #         self.set_value_from_xy_index(ix - 1, iy - 1, val=1.0)

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):

        npoint = len(x) - 1
        inside = False
        for i1 in range(npoint):
            i2 = (i1 + 1) % (npoint + 1)

            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]

            if not min_x < iox < max_x:
                continue

            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside

        return inside

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("num_grid:", self.num_grid)

    def plot_grid_map(self, ax=None):
        # grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(self.grid_map, cmap="jet", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map

    def plot_grid_map_in_space(self, emph=None, alpha=0.5, zorder=0, ax=None):
        # grid_data = np.reshape(np.array(self.data), (self.height, self.width)).astype(np.int)
        if not ax:
            fig, ax = plt.subplots()
        extent = [self.min_x, self.max_x, self.min_y, self.max_y]

        if emph is None:
            ax.imshow(self.grid_map, origin='lower', alpha=alpha, extent=extent, cmap="Blues", zorder=zorder)
        else:
            ax.imshow(self.grid_map, origin='lower', alpha=alpha, extent=extent, cmap="Blues", zorder=zorder)
            ax.imshow(emph, origin='lower', alpha=0.5, extent=extent, cmap="Reds", zorder=zorder)

        ax.axis("equal")

def test_polygon_set():
    ox0 = [30.0, 50.0, 80.0, 130.0, 160.0, 70.0]
    oy0 = [30.0, 10.0, 30.0, 60.0, 90.0, 110.0]


    ox1 = [5.0, 5.0, 15.0, 15.0]
    oy1 = [5.0, 15.0, 15.0, 5.0]


    ox2 = [0.0, 0.0, 20.0, 20.0]
    oy2 = [20.0, 40.0, 40.0, 20.0]


    ox3 = [50.0, 50.0, 80.0, 80.0]
    oy3 = [-20.0, -40.0, -40.0, -20.0]

    grid_map = RectangularGridMap(600, 290, 0.7, 60.0, 30.5)

    grid_map.set_value_from_polygon(ox0, oy0, 1.0, inside=True)
    grid_map.set_value_from_polygon(ox1, oy1, 1.0, inside=True)
    grid_map.set_value_from_polygon(ox2, oy2, 1.0, inside=True)
    grid_map.set_value_from_polygon(ox3, oy3, 1.0, inside=True)

    grid_map.plot_grid_map_in_space()
    #
    plt.axis("equal")
    plt.grid(True)

    return grid_map


def test_position_set():
    grid_map = RectangularGridMap(100, 120, 0.5, 10.0, -0.5)

    grid_map.set_value_from_xy_pos(10.1, -1.1, 1.0)
    grid_map.set_value_from_xy_pos(10.1, -0.1, 1.0)
    grid_map.set_value_from_xy_pos(10.1, 1.1, 1.0)
    grid_map.set_value_from_xy_pos(11.1, 0.1, 1.0)
    grid_map.set_value_from_xy_pos(10.1, 0.1, 1.0)
    grid_map.set_value_from_xy_pos(9.1, 0.1, 1.0)

    grid_map.plot_grid_map()


def main():
    print("start!!")
    gmap = test_polygon_set()

    egmap = test_extract(gmap)

    # test_position_set()
    # test_polygon_set()

    plt.show()

    print("done!!")


if __name__ == '__main__':
    main()