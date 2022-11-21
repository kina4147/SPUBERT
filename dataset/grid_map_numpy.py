import numpy as np
import math


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

        self.min_x = self.center_x - self.width / 2.0 * self.resolution
        self.min_y = self.center_y - self.height / 2.0 * self.resolution
        self.max_x = self.center_x + self.width / 2.0 * self.resolution
        self.max_y = self.center_y + self.height / 2.0 * self.resolution

        self.num_grid = self.width * self.height
        self.grid_map = np.full((self.height, self.width), init_val)

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
        self.grid_map[y_ids, x_ids] = vals

    def flip_x(self):
        self.grid_map = np.flip(self.grid_map, axis=0)

    def flip_y(self):
        self.grid_map = np.flip(self.grid_map, axis=1)

    def rotate(self, theta):
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
        x_ids, y_ids = self.get_all_indices()
        x_pos, y_pos = self.calc_grid_center_xy_position_from_xy_index(x_ids, y_ids)
        for x_idx, y_idx, x, y in zip(x_ids, y_ids, x_pos, y_pos):
            flag = self.check_inside_polygon(x, y, pol_x, pol_y)
            if flag is inside:
                self.set_value_from_xy_index(x_idx, y_idx, val)

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
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(self.grid_map, cmap="jet", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map

    def plot_grid_map_in_space(self, emph=None, alpha=0.5, zorder=0, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        extent = [self.min_x, self.max_x, self.min_y, self.max_y]

        if emph is None:
            ax.imshow(self.grid_map, origin='lower', alpha=alpha, extent=extent, cmap="Blues", zorder=zorder)
        else:
            ax.imshow(self.grid_map, origin='lower', alpha=alpha, extent=extent, cmap="Blues", zorder=zorder)
            ax.imshow(emph, origin='lower', alpha=0.5, extent=extent, cmap="Reds", zorder=zorder)

        ax.axis("equal")
