import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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

def accuracy_by_treedepth(features,targets,depths):
    features_train, features_test = features[:features.shape[0]//2], features[features.shape[0]//2:]
    targets_train, targets_test = targets[:targets.shape[0]//2], targets[targets.shape[0]//2:]
    diffs_validation=[]
    diffs_training=[]
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(features_train, targets_train)
        model_predictions = model.predict(features_test)
        diffs_validation.append(median_diff(model_predictions, targets_test))
        model_predictions_train = model.predict(features_train)
        diffs_training.append(median_diff(model_predictions_train, targets_train))
    return diffs_validation, diffs_training

data = np.load(r'Module 5\5a\sdss_galaxy_colors.npy')
features,targets=get_features_targets(data)
depths = np.arange(1, 51)
diffs_validation, diffs_training = accuracy_by_treedepth(features, targets, depths)
plt.plot(depths, diffs_validation, label='Validation set')
plt.plot(depths, diffs_training, label='Training set')
plt.ylabel("Median of Differences")
plt.xlabel("Maximum Tree Depth")
plt.legend()
plt.show()