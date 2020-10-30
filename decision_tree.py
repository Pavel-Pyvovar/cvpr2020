from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

image_descriptions_df = pd.read_pickle('./results/image_descriptions.pkl')

model = DecisionTreeClassifier()

pca = PCA(n_components=3)
sample0 = image_descriptions_df.descriptor_values[0]
new_sample0 = pca.fit_transform(sample0)
print(new_sample0.shape)