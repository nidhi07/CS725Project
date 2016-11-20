import numpy
import xgboost as xgb
import pandas as pd 
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import csv
from matplotlib.pylab import rcParams
import numpy as np

# load data
# X = numpy.loadtxt('X_train.csv', delimiter=",")
# Y = numpy.loadtxt('y_train.csv', delimiter=",")


def oneHotEncoding(preds):
    oneHotClass  = []
    for p in preds:
        oneHotClass.append(np.argmax(p))
    return oneHotClass

X = pd.read_csv('data/X_train.csv' , header = 0)
Y = pd.read_csv('data/y_train.csv' , header = 0)
X2 = pd.read_csv('data/X_test.csv' , header = 0)
Y2 = pd.read_csv('data/y_test.csv' , header = 0)
Y = Y.iloc[0::,1]
Y2 = Y2.iloc[0::,1]
#print("type(X) " , type(X))
#print("type(Y) " , type(Y))
#print("X.shape",X.shape)
#print("Y.shape",Y.shape)


#split data into train and test sets
seed = 5
test_size = 0.33

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
print(X_train.shape)
print(X_test.shape)

xgb_params = {
    "objective": "multi:softprob",
    "eta": 0.1,
    "num_class": 3,
    "max_depth": 6,
    "nthread": 4,
    "eval_metric": "mlogloss",
    "print.every.n": 1
    #"silent": 1
}
num_rounds = 4

mask = np.random.choice([False, True], len(X_train), p=[0.80, 0.20])
not_mask = [not i for i in mask]

dtrain = xgb.DMatrix(X_train[not_mask], label=y_train[not_mask])
dtrain_watch = xgb.DMatrix(X_train[mask], label=y_train[mask])
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train'),(dtrain_watch, 'test')]
model = xgb.train(xgb_params, dtrain, num_rounds, watchlist)


preds = model.predict(dtest)
predictions = oneHotEncoding(preds)
# evaluate predictions on our test set got from cross validation
test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy using CV: %.2f%%" % (test_accuracy * 100.0))


#print accuracy on training set that was used for training -it was obtained from cross validation
#use complete train - dont mask it
dtrain_accuracy = xgb.DMatrix(X_train)
preds_train = model.predict(dtrain_accuracy)
predictions_train = oneHotEncoding(preds_train)
# evaluate predictions on our test set got from cross validation
train_accuracy = accuracy_score(y_train, predictions_train)
print("Train Accuracy on whole train set: %.2f%%" % (train_accuracy * 100.0))


#Predict for actual test file and print it in output6.csv
dtest_actual = xgb.DMatrix(X2)
preds_test = model.predict(dtest_actual)
#predictions_test = [round(value) for value in preds_test]
#write to csv file
with open('xgb_output.csv', 'w') as csvfile:
    output_writer = csv.writer(csvfile, delimiter=',')
    output_writer.writerow(['id','predict_0','predict_1','predict_2'])
    for i in range(0,len(X2)):
        predict0 = preds_test[i][0]
        predict1 = preds_test[i][1]
        predict2 = preds_test[i][2]
        output_writer.writerow([X2['id'][i],predict0,predict1,predict2])

