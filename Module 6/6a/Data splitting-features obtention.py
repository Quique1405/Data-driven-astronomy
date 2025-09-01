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
