import numpy as np
from scipy import linalg
from statistics import NormalDist
from tqdm import tqdm


class InvalidData(Exception):
    "Exception raised when data is incompatible with the Norta method"


class Norta:

    "Class to compute the Norta method (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.281&rep=rep1&type=pdf)"

    def __init__(self, samples: np.array) -> None:
        """Init method for Norta class. In this method, the input
        data are validated and the covariance matrices and the
        corresponding lower triangular decomposition are computed.

        Parameters
        ----------
        samples : np.array
            Array containing the observations from which the samples
            will be generated. The shape required is (rows, cols):
                -rows -> Number of observations
                -columns -> Number of variables
        """

        if len(samples.shape) != 2:
            raise InvalidData(
                "Input array does not match the shape requirements. It must be a two dimensional array"
            )

        self.samples = samples
        self.cov_m = np.cov(samples.T)
        _, self.lower_triangular, _ = linalg.lu(self.cov_m)

    def generate_samples(self, n_samples: int, n_bins: int=None, verbose: int = 1) -> np.array:
        """Main method to generate samples from the given distribution via NORTA method.

        Parameters
        ----------
        n_samples : int
            Number of new samples to be generated
        n_bins : int, optional
            Number of bins to generate the cumulative distribution functions. If not
            provided it will be set as <number_of_observations>//20, by default None
        verbose : int, optional
            Verbosity lebel, by default 1:
                -0 -> The progress bars (tqdm) WILL NOT be displayed on the console.
                -1 -> The progress bars (tqdm) WILL be displayed on the console.

        Returns
        -------
        np.array
            Generated samples via NORTA method with shape (<number of observations>, <number of variables> )
        """

        standard_normal = NormalDist(mu=0, sigma=1.0)

        verbosity = not bool(verbose)

        if n_bins is None:
            n_bins = self.samples.shape[0] // 20

        generated_samples = np.zeros((n_samples, self.cov_m.shape[0]))

        densities = np.zeros((self.cov_m.shape[0], n_bins))
        bins = np.zeros((self.cov_m.shape[0], n_bins + 1))
        cdfs = np.zeros((self.cov_m.shape[0], n_bins))

        for i in tqdm(range(self.cov_m.shape[0]), disable=verbosity):
            densities[i, :], bins[i, :] = np.histogram(
                self.samples[:, i], bins=n_bins, density=True
            )

            dist = bins[i, 1] - bins[i, 0]
            cdfs[i, :] = np.cumsum(densities[i, :]) * dist

        for n in tqdm(range(n_samples), disable = verbosity):
            W = np.random.normal(
                size=(self.lower_triangular.shape[0], 1), loc=0, scale=1
            )
            Z = self.lower_triangular @ W

            generated_sample = np.zeros((self.lower_triangular.shape[0], 1))
            for i in range(Z.shape[0]):
                generated_sample[i] = self.inverse_cdf(
                    cdfs[i, :], bins[i, :], standard_normal.cdf(Z[i])
                )
            generated_samples[n, :] = generated_sample.squeeze(-1)

        return generated_samples

    @staticmethod
    def inverse_cdf(cdf: np.array, bins: np.array, value: float) -> float:
        
        """Static method to compute the inverse cdf from
        a given discrete cdf and its corresponding bins (x coordinates 
        of the given discrete cdf).

        Parameters
        ----------
        cdf : np.array
            Discrete cumulative distribution function of the variable.
        bins : np.array
            X coordinates for the cdf.
        value : float
            Value for which to calculate the inverse cdf.

        Returns
        -------
        float
            Inverse cdf of the given value.
        """

        distances = np.abs(cdf - value)
        indexes = np.argpartition(distances, 2)[:2]

        out_value = (
            bins[indexes[1]] * distances[indexes[0]]
            + bins[indexes[0]] * distances[indexes[1]]
        ) / (distances[indexes[0]] + distances[indexes[1]])

        return out_value
