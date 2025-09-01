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
def cross_validate_predictions(model, features, targets, k):
    kf = KFold(n_splits=k, shuffle=True)

    # declare an array for predicted redshifts from each iteration
    all_predictions = np.zeros(shape=len(targets))

    for train_indices, test_indices in kf.split(features):
        # split the data into training and testing
        train_features, test_features = features[train_indices], features[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]
    
        # fit the model for the current set
        model.fit(train_features, train_targets)
        # predict using the model
        predictions = model.predict(test_features)
        # put the predicted values in the all_predictions array defined above
        all_predictions[test_indices] = predictions

    # return the predictions
    return all_predictions
data = np.load(r'Module 5\5a\sdss_qso_colors.npy')
features, targets = get_features_targets(data) 
