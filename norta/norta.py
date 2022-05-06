import numpy as np
from scipy import linalg
from statistics import NormalDist
from tqdm import tqdm

class Norta:
    def __init__(self, samples):
        self.samples = samples
        self.cov_m = self._compute_cov_matrix(self.samples)

        self._compute_cholesky()

    def _compute_cov_matrix(self, samples):
        return np.cov(samples)

    def _compute_cholesky(self):
        _, self.lower_triangular, _ = linalg.lu(self.cov_m)

    def generate_samples(self, n_samples, n_bins=None, verify=False):
        standard_normal = NormalDist(mu=0, sigma=1.0)

        if n_bins is None:
            n_bins = self.samples.shape[1] // 20

        generated_samples = np.zeros((n_samples, self.cov_m.shape[0]))

        densities = np.zeros((self.cov_m.shape[0], n_bins))
        bins = np.zeros((self.cov_m.shape[0], n_bins + 1))
        cdfs = np.zeros((self.cov_m.shape[0], n_bins))

        for i in tqdm(range(self.cov_m.shape[0])):
            densities[i, :], bins[i, :] = np.histogram(
                self.samples[i, :], bins=n_bins, density=True
            )

            dist = bins[i, 1] - bins[i, 0]
            cdfs[i, :] = np.cumsum(densities[i, :]) * dist 


        for n in tqdm(range(n_samples)):
            W = np.random.normal(size=(self.lower_triangular.shape[0], 1), loc = 0, scale= 1)
            Z = self.lower_triangular @ W

            generated_sample = np.zeros((self.lower_triangular.shape[0], 1))
            for i in range(Z.shape[0]):
                generated_sample[i] = self.inverse_cdf(
                    cdfs[i, :], bins[i, :], standard_normal.cdf(Z[i])
                )
            generated_samples[n, :] = generated_sample.squeeze(-1)

        return generated_samples

    @staticmethod
    def inverse_cdf(cdf, bins, value):
        distances = np.abs(cdf - value)
        indexes = np.argpartition(distances, 2)[:2]

        value = (
            bins[indexes[1]] * distances[indexes[0]]
            + bins[indexes[0]] * distances[indexes[1]]
        ) / (distances[indexes[0]] + distances[indexes[1]])
        
        return value
