"""
Write a get_features_targets function that splits the training data into input features and their corresponding targets. In our case, the inputs are the 4 colour indices and our targets are the corresponding redshifts.

Your function should return a tuple of:

features: a NumPy array of dimensions m ⨉ 4, where m is the number of galaxies;
targets: a 1D NumPy array of length m, containing the redshift for each galaxy.
The data argument will be the structured array described on the previous slide. The u flux magnitudes and redshifts can be accessed as a column with data['u'] and data['redshift'].
"""
import numpy as np
data = np.load(r"Module 5\sdss_galaxy_colors.npy")
print(len(data))
def get_features_targets(data):
    u_g = data['u']-data['g']
    g_r = data['g']-data['r']
    r_i = data['r']-data['i']
    features = np.vstack((u_g, g_r, r_i))
    targets = data['redshift']
    return features.transpose(), targets
features, targets = get_features_targets(data)
print(features.shape, targets.shape)
