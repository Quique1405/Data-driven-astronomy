import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
def plot_confusion_matrix(cm, classes,  
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):   #Provided by the course resources
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

def generate_features_targets(data):
    features=np.empty(shape=(len(data),13))
    for i in range(10):
        features[:,i]=data[data.dtype.names[i]]
    for j in range(10,13):
        features[:,j]=data[data.dtype.names[j]]/data[data.dtype.names[j+3]]
    targets=data['class']
    return features,targets

def calculate_accuracy(predicted, actual):
  npredicted=(predicted==actual).astype(float).sum()
  nactual=len(predicted)
  return npredicted/nactual

def splitdata_train_test(data,fraction):
    indi=int(len(data)*fraction)
    training_set,testing_set=data[:indi],data[indi:]
    return training_set,testing_set

def rf_predict_actual(data, n_estimators):
    np.random.shuffle(data)
  # generate the features and targets
    features,targets=generate_features_targets(data)
  # instantiate a random forest classifier using n estimators
    dtc=RandomForestClassifier(n_estimators=n_estimators)
    dtc.fit(features, targets)
  # get predictions using 10-fold cross validation with cross_val_predict
    predictions = cross_val_predict(dtc, features, targets, cv=10)
  # return the predictions and their actual classes
    return predictions,targets