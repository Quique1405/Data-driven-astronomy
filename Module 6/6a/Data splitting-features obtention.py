"""
Your task is to implement the splitdata_train_test function.
It takes a NumPy array and splits it into a training and testing NumPy array 
based on the specified training fraction. 
The function takes two arguments and should return two values:

Arguments
data: the NumPy array containing the galaxies in the form described in the previous slide;
fraction_training: the fraction of the data to use for training. This will be a float between 0 and 1.

Return values
training_set: the first value is a NumPy array training set;
testing_set: the second value is a NumPy array testing set.

"""
import numpy as np
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
data=np.load(r"Module 6\galaxy_catalogue.npy")
train_set,test_set=splitdata_train_test(data,0.7)
features, targets =generate_features_targets(data)
print(data.dtype.names)
print((data[data.dtype.names[10]]/data[data.dtype.names[10+3]])[0])
concu=data[data.dtype.names[10]]/data[data.dtype.names[10+3]]
print(concu[0])
print(features[0])