import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets
def median_diff(predicted, actual):
  return np.median(np.abs(predicted - actual))
# complete this function
def cross_validate_model(model, features, targets, k):
    kf = KFold(n_splits=k, shuffle=True)
    # initialise a list to collect median_diffs for each iteration of the loop below
    median_diffs = []
    for train_indices, test_indices in kf.split(features):
        train_features, test_features = features[train_indices], features[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]
        # fit the model for the current set
        model.fit(train_features, train_targets)
        # predict using the model
        predictions=model.predict(test_features)
        # calculate the median_diff from predicted values and append to results array
        median_diffs.append(median_diff(predictions, test_targets))
    # return the list with your median difference values
    return median_diffs
def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  Spec=data['spec_class']
  galaxies=data[Spec==b'GALAXY']
  qsos=data[Spec==b'QSO']
  return galaxies,qsos
def cross_validate_median_diff(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return np.mean(cross_validate_model(dtr, features, targets, 10))

data = np.load(r'Module 5\5a\sdss_galaxy_colors.npy')

galaxies, qsos= split_galaxies_qsos(data)
galaxy_med_diff = cross_validate_median_diff(galaxies)
qso_med_diff = cross_validate_median_diff(qsos)
print("Median difference for Galaxies: {:.3f}".format(galaxy_med_diff))
print("Median difference for QSOs: {:.3f}".format(qso_med_diff))