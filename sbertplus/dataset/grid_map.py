"""

Grid map library in python

author: Atsushi Sakai

"""

import matplotlib.pyplot as plt
import numpy as np
import math

# 전체 그리드를 각도에 따라서 다 갖고 있고, 이들이 거리가, 해당 점의 각도로 각도 인덱스를 뽑아내고, 거리가 가까운 것을 식별한다.
# 각도 인덱스에 그리드맵의 모든 셀을 매핑시킨다.

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
    for x, y in zip(xs, ys):
        src_val = src_gmap.get_value_from_xy_pos(x, y)
        dst_gmap.set_value_from_xy_pos(x, y, src_val)

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

# def rotate_grid_map(theta):
#     # all dst x, y pos
#
#     # all rotate dst x y pos to src x, y pos
#
#     # get value from src x y pos
#
#     # projection
#
#     # d = math.hypot(x, y)
#     # rx = d * np.cos(theta)
#     # ry = d * np.sin(theta)
#
#
# def rotate_cgrid_map(theta):
#     # all dst d a pos
#
#     # rotate ? a += theta
#
#     # get value from src da pos
#
#     # projection


class CircularGridMap:
    def __init__(self, width, height, d_resol, a_resol, center_x, center_y, center_d, center_a, init_val=0, ray_casting=False):

        self.width = width
        self.d_resol = d_resol
        self.center_d = center_d

        self.height = height
        self.a_resol = a_resol
        self.center_a = center_a
        self.init_val = init_val

        self.center_x = center_x
        self.center_y = center_y

        self.min_d = 0 # self.center_x - self.width * self.d_resol
        self.min_a = self.center_a - self.height / 2.0 * self.a_resol
        self.max_d = self.width * self.d_resol
        self.max_a = self.center_a + self.height / 2.0 * self.a_resol

        self.ndata = self.width * self.height
        self.data = [self.init_val] * self.ndata
        self.ray_casting = ray_casting

    def calc_xy_pos_from_da_pos(self, d_pos, a_pos):
        y_pos = self.center_y + d_pos * np.sin(a_pos)
        x_pos = self.center_x + d_pos * np.cos(a_pos)
        return x_pos, y_pos

    def get_all_xy_pos(self):
        d_pos, a_pos = self.get_all_da_pos()
        x_pos, y_pos = self.calc_xy_pos_from_da_pos(d_pos, a_pos)
        return x_pos, y_pos

    def get_all_da_pos(self):
        d_idx, a_idx = self.get_all_da_idx()
        d_pos, a_pos = self.calc_da_pos_from_da_idx(d_idx, a_idx)
        return d_pos, a_pos

    def get_all_da_idx(self):
        d_idx, a_idx = np.mgrid[slice(0, self.width, 1),
                                slice(0, self.height, 1)]
        return d_idx.flatten(), a_idx.flatten()


    def get_da_idx_from_xy_pos(self, x, y):
        d_pos, a_pos = self.calc_da_pos_from_xy_pos(x, y)
        d_idx = self.calc_idx_from_pos(d_pos, self.min_d, self.d_resol, self.width)
        a_idx = self.calc_idx_from_pos(a_pos, self.min_a, self.a_resol, self.height)
        if (not d_idx) or (not a_idx):
            return False, False
        return d_idx, a_idx

    def calc_da_pos_from_xy_pos(self, x, y):
        x -= self.center_x
        y -= self.center_y
        d = math.hypot(x, y)
        a = math.atan2(y, x)
        return d, a

    def update_ray_casting(self):
        # rays = []
        for a_idx in range(self.height):
            occupied = False
            for d_idx in range(self.width):
                if occupied:
                    self.set_value_from_da_index(d_idx, a_idx, self.init_val)
                val = self.get_val_from_da_idx(d_idx, a_idx)
                if val != self.init_val:
                    occupied = True
                    # rays.append(val)

    def get_val_from_da_idx(self, d_idx, a_idx):
        """get_val_from_da_idx

        when the index is out of grid map area, return None

        :param d_idx: d index
        :param a_idx: a index
        """
        grid_ind = self.calc_grid_idx_from_da_idx(d_idx, a_idx)
        if 0 <= grid_ind < self.ndata:
            return self.data[grid_ind]
        else:
            return None

    def update_val_from_xy_pos(self, x_pos, y_pos, up_val):

        """update_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param up_val: update grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        if (not x_ind) or (not y_ind):
            return False  # NG
        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)
        self.data[grid_ind] += up_val
        return True

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        d_idx, a_idx = self.get_da_idx_from_xy_pos(x_pos, y_pos)
        if (not d_idx) or (not a_idx):
            return False
        flag = self.set_value_from_da_index(d_idx, a_idx, val)
        return flag

    def set_value_from_da_index(self, d_idx, a_idx, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (d_idx is None) or (a_idx is None):
            return False

        grid_ind = int(a_idx * self.width + d_idx)

        if 0 <= grid_ind < self.ndata:
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def calc_grid_idx_from_da_idx(self, d_idx, a_idx):
        grid_ind = int(a_idx * self.width + d_idx)
        return grid_ind

    def calc_da_pos_from_da_idx(self, d_ind, a_ind):
        d_pos = self.calc_pos_from_idx(d_ind, self.min_d, self.d_resol)
        a_pos = self.calc_pos_from_idx(a_ind, self.min_a, self.a_resol)
        return d_pos, a_pos

    def calc_pos_from_idx(self, idx, lower_pos, resol):
        return lower_pos + idx * resol + resol / 2.0

    def calc_idx_from_pos(self, pos, lower_pos, resol, max_index):
        ind = int(np.floor((pos - lower_pos) / resol))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def plot_grid_map(self, ax=None):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map

    def plot_grid_map_in_space(self, ax=None):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()

        extent = [self.min_d, self.max_d, self.min_a, self.max_a]
        plt.imshow(grid_data, origin='lower', extent=extent, cmap="Blues")
        plt.axis("auto")
        plt.ylim([self.min_a, self.max_a])
        plt.xlim([self.min_d, self.max_d])



class RectangularGridMap:
    """
    GridMap class
    """

    def __init__(self, width, height, resolution,
                 center_x, center_y, init_val=0.0):
        """__init__

        :param width: number of grid for width
        :param height: number of grid for height
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

        self.min_x = self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        self.min_y = self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution
        self.max_x = self.center_x + self.width / 2.0 * self.resolution
        self.max_y = self.center_y + self.height / 2.0 * self.resolution

        self.ndata = self.width * self.height
        self.data = [init_val] * self.ndata

    def get_all_positions(self):
        # x, y = np.mgrid[slice(self.min_x - self.resolution / 2.0, self.max_x + self.resolution / 2.0, self.resolution),
        #                 slice(self.min_y - self.resolution / 2.0, self.max_y + self.resolution / 2.0, self.resolution)]

        x_ind, y_ind = self.get_all_indices()
        x = self.calc_grid_central_xy_position_from_index(x_ind, self.left_lower_x)
        y = self.calc_grid_central_xy_position_from_index(y_ind, self.left_lower_y)
        return x, y

    def get_all_indices(self):
        x_ind, y_ind = np.mgrid[slice(0, self.height, 1),
                                slice(0, self.width, 1)]
        return x_ind.flatten(), y_ind.flatten()


    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index

        when the index is out of grid map area, return None

        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.ndata:
            return self.data[grid_ind]
        else:
            return None

    def update_val_from_xy_pos(self, x_pos, y_pos, up_val):

        """update_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param up_val: update grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        if (not x_ind) or (not y_ind):
            return False  # NG
        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)
        self.data[grid_ind] += up_val
        return True

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(
            x_pos, self.left_lower_x, self.width, self.center_x)
        y_ind = self.calc_xy_index_from_position(
            y_pos, self.left_lower_y, self.height, self.center_y)

        return x_ind, y_ind

    def get_value_from_xy_pos(self, x_pos, y_pos):

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)
        if (not x_ind) or (not y_ind):
            return self.init_val
        return self.get_value_from_xy_index(x_ind, y_ind)
        # grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)
        # return self.data[grid_ind]

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if (not x_ind) or (not y_ind):
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False, False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.ndata:
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def flip_x(self):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        grid_data = np.flip(grid_data, axis=0)
        self.data = grid_data.flatten().tolist()

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
        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(
                    x_ind, y_ind)

                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)

                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(
            x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(
            y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos, max_index, center):
        ind = int(np.floor((pos - lower_pos) / self.resolution))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def check_occupied_from_xy_index(self, xind, yind, occupied_val=1.0):

        val = self.get_value_from_xy_index(xind, yind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self):
        xinds, yinds = [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy):
                    xinds.append(ix)
                    yinds.append(iy)

        for (ix, iy) in zip(xinds, yinds):
            self.set_value_from_xy_index(ix + 1, iy, val=1.0)
            self.set_value_from_xy_index(ix, iy + 1, val=1.0)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=1.0)
            self.set_value_from_xy_index(ix - 1, iy, val=1.0)
            self.set_value_from_xy_index(ix, iy - 1, val=1.0)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=1.0)

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
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("ndata:", self.ndata)

    def plot_grid_map(self, ax=None):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="jet", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map

    def plot_grid_map_in_space(self, alpha=None, ax=None):
        grid_data = np.reshape(np.array(self.data), (self.height, self.width)).astype(np.int)
        if not ax:
            fig, ax = plt.subplots()
        extent = [self.min_x, self.max_x, self.min_y, self.max_y]
        if alpha is None:
            ax.imshow(grid_data, origin='lower', extent=extent, cmap="Blues")
        else:
            ax.imshow(grid_data, origin='lower', extent=extent, cmap="Blues")
            ax.imshow(alpha, origin='lower', alpha=0.2, extent=extent, cmap="Reds")

        plt.axis("equal")

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

    # grid_map.plot_grid_map_in_space()
    #
    # plt.axis("equal")
    # plt.grid(True)

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