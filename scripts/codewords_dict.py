import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from utils import read_results

# Constants
N_CLUSTERS = 10

descriptions = read_results('descriptions.pkl')
data = descriptions.decriptor_values.to_numpy()
X = np.vstack(data)

k_means = cluster.KMeans(N_CLUSTERS)
k_means.fit(X)
print(X.shape)
preds = k_means.predict(X)
print(preds.shape)
