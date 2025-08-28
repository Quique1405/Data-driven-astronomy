import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

def plot_confusion_matrix(cm, classes,  
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):   #Provided by the course recourses
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

data=np.load(r"Module 6\galaxy_catalogue.npy")

  # split the data
features, targets = generate_features_targets(data)

  # train the model to get predicted and actual classes
dtc = DecisionTreeClassifier()
predicted = cross_val_predict(dtc, features, targets, cv=10)

  # calculate the model score using your function
model_score = calculate_accuracy(predicted, targets)
print("Our accuracy score:", model_score)

  # calculate the models confusion matrix using sklearns confusion_matrix function
class_labels = list(set(targets))
model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

  # Plot the confusion matrix using the provided functions.
plt.figure()
plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
plt.show()