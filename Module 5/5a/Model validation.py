import numpy as np
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

def validate_model(model, features, targets):
  split = features.shape[0]//2
  train_features, test_features = features[:split], features[split:]
  train_targets, test_targets = targets[:split], targets[split:]
  model.fit(train_features, train_targets)
  predictions = model.predict(test_features)  
  return median_diff(test_targets, predictions)

data = np.load(r'Module 5\5a\sdss_galaxy_colors.npy')
features, targets = get_features_targets(data)
dtr = DecisionTreeRegressor()
diff = validate_model(dtr, features, targets)
print('Median difference: {:f}'.format(diff))