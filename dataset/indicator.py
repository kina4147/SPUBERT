

# class Spatial6DIndicator(object):
#     def __init__(self, bound_range=20.0):
#
#         self.pad_id = [-bound_range, -bound_range, -bound_range, -bound_range, -bound_range, -bound_range]
#         self.msk_id = [ bound_range,  bound_range, bound_range, bound_range, bound_range, bound_range]  # unknown index should be the same
#         self.sep_id = [ bound_range, -bound_range, bound_range, bound_range, bound_range, bound_range]
#         self.sot_id = [-bound_range,  bound_range, bound_range, bound_range, bound_range, bound_range]
#         self.false_id = [0, 0, 0, 0, 0, 0]  #[False, False]
#         self.true_id  = [1, 1, 1, 1, 1, 1]  #[ True,  True]
#         self.msk_val = bound_range
#         self.pad_val = -bound_range
#         self.false_val = 0
#         self.true_val = 1
#
# class Spatial4DIndicator(object):
#     def __init__(self, bound_range=20.0):
#         self.pad_id = [-bound_range, -bound_range, 0, 0]
#         self.msk_id = [ bound_range,  bound_range, 0, 0]  # unknown index should be the same
#         self.sep_id = [ bound_range, -bound_range, 0, 0]
#         self.sot_id = [-bound_range,  bound_range, 0, 0]
#         self.false_id = [0, 0, 0, 0]  #[False, False]
#         self.true_id  = [1, 1, 1, 1]  #[ True,  True]
#         self.msk_val = bound_range
#         self.pad_val = -bound_range
#         self.false_val = 0
#         self.true_val = 1


class SpatialIndicator(object):
    def __init__(self, bound_range=20.0):
        self.pad_id = [-bound_range, -bound_range]
        self.msk_id = [ bound_range,  bound_range]  # unknown index should be the same
        self.sep_id = [ bound_range, -bound_range]
        self.sot_id = [-bound_range,  bound_range]
        self.false_id = [0, 0]  #[False, False]
        self.true_id  = [1, 1]  #[ True,  True]
        self.msk_val = bound_range
        self.pad_val = -bound_range
        self.false_val = 0
        self.true_val = 1

class IntegerIndicator(object):
    def __init__(self):
        self.pad_id = 0
        self.msk_id = 1
        self.sep_id = 2
        self.sot_id = 3

        self.false_val = 0
        self.true_val = 1

        self.none_lbl = 0
        self.far_lbl = 1
        self.near_lbl = 2