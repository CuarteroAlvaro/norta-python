import matplotlib.pyplot as plt
from norta import Norta 
import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = np.zeros((100000, 2))
    data[:, 0] = np.random.normal(size = data.shape[0], loc = 0, scale = 17)
    data[:, 1] = data[:, 0] + np.random.normal(size = data.shape[0], loc = 6, scale = 1)

    norta = Norta(data)
    samples = norta.generate_samples(n_samples=100000, n_bins = 150)

    for i in range(data.shape[1]):
        df = pd.DataFrame({'original':  data[:, i], 'generated': samples[:, i]})

        df['original'].hist(bins = 100, density = 1, legend = "original", alpha = 0.4)
        df['generated'].hist(bins = 100, density = 1, legend = "generated", alpha = 0.4)

        plt.show()
        plt.close()
