import numpy as np
from sklearn.tree import DecisionTreeClassifier
def splitdata_train_test(data,fraction):
    indi=int(len(data)*fraction)
    training_set,testing_set=data[:indi],data[indi:]
    return training_set,testing_set

def generate_features_targets(data):
    features=np.empty(shape=(len(data),13))
    for i in range(10):
        features[:,i]=data[data.dtype.names[i]]
    for j in range(10,13):
        features[:,j]=data[data.dtype.names[j]]/data[data.dtype.names[j+3]]
    targets=data['class']
    return features,targets

# complete this function by splitting the data set and training a decision tree classifier
def dtc_predict_actual(data):
  # split the data into training and testing sets using a training fraction of 0.7
    training_set,testing_set=splitdata_train_test(data,0.7)
  # generate the feature and targets for the training and test sets
  # i.e. train_features, train_targets, test_features, test_targets
    features_training,targets_training=generate_features_targets(training_set)
    features_test,targets_test=generate_features_targets(testing_set)
  # instantiate a decision tree classifier
    dtr=DecisionTreeClassifier()
  # train the classifier with the train_features and train_targets
    dtr.fit(features_training,targets_training)
  # get predictions for the test_features
    predictions=dtr.predict(features_test)
  # return the predictions and the test_targets
    return predictions,targets_test
data=np.load(r"Module 6\galaxy_catalogue.npy")
np.random.shuffle(data)
predicted_class, actual_class = dtc_predict_actual(data)
for i in range(10):
    print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))