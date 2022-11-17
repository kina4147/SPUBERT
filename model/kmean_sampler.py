import torch
import numpy as np

class MultiKMeans:
    def __init__(self, n_clusters, n_kmeans, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", minibatch=None):
        self.n_clusters = n_clusters
        self.n_kmeans = n_kmeans
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.minibatch = minibatch
        # self._loop = False
        self._show = True

        try:
            import PYNVML
            self._pynvml_exist = True
        except ModuleNotFoundError:
            self._pynvml_exist = False

        self.centroids = None
        self.num_points_in_clusters = None

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        return 2 * a @ b.transpose(-2, -1) - (a ** 2).sum(dim=-1)[..., :, None] - (b ** 2).sum(dim=-1)[..., None, :]

    def remaining_memory(self):
        """
          Get remaining memory in gpu
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self._pynvml_exist:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        else:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def max_sim(self, a, b):
        """
          Compute maximum similarity (or minimum distance) of each vector
          in a with all of the vectors in b
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        # device = a.device.type
        # batch_size = a.shape[-2]
        if self.mode == 'cosine':
            sim_func = self.cos_sim
        elif self.mode == 'euclidean':
            sim_func = self.euc_sim
        sim = sim_func(a, b)
        max_sim_v, max_sim_i = sim.max(dim=-1)
        return max_sim_v, max_sim_i

    def fit_predict(self, X):
        """
          Combination of fit() and predict() methods.
          This is faster than calling fit() and predict() seperately.
          Parameters:
          X: torch.Tensor, shape: [n_samples, n_features]
          centroids: {torch.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X
          Return:
          labels: torch.Tensor, shape: [n_samples]
        """
        n_stream, batch_size, emb_dim = X.shape
        device = X.device.type
        self.centroids = X[:, np.random.choice(batch_size, size=[self.n_clusters]), :]

        self.num_points_in_clusters = torch.ones(self.n_kmeans, self.n_clusters, device=device)
        # closest = None
        for i in range(self.max_iter):
            closest = self.max_sim(a=X, b=self.centroids)[1]
            # matched_clusters, counts = closest.unique(return_counts=True)
            uniques = [closest[i].unique(return_counts=True) for i in range(self.n_kmeans)]
            # c_grad = torch.zeros_like(self.centroids)
            expanded_closest = closest[:, None].expand(-1, self.n_clusters, -1)
            mask = (expanded_closest == torch.arange(self.n_clusters, device=device)[None, :, None]).float()
            c_grad = mask @ X / mask.sum(-1, keepdim=True)
            c_grad[c_grad != c_grad] = 0  # remove NaNs
            error = (c_grad - self.centroids).pow(2).sum()
            for j in range(self.n_kmeans):
                self.num_points_in_clusters[j, uniques[j][0]] += uniques[j][1]
            self.centroids = c_grad
            if error <= self.tol * self.n_kmeans:
                break

        return self.centroids
