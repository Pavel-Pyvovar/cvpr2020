import cv2
import numpy as np
import pandas as pd
import pickle
import os
import scipy as sp
from scipy.misc import face
from sklearn import cluster

face = face(gray=True)
print(face.shape)
X = face.reshape((-1, 1))
print(X.shape)

results_path = f'{os.path.dirname(os.path.realpath(__file__))}/../results'
results = pd.read_pickle(f'{results_path}/descriptions.pkl')
# print(results)
