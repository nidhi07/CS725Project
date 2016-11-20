import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

def predict_rf():
	X= pd.read_csv('data/X_train.csv',header=0)
	y= pd.read_csv('data/y_train.csv',header=0)
	X= X.drop(['id'],axis=1)
	y=y['fault_severity']


	testX= pd.read_csv('data/X_test.csv',header=0)
	testY= pd.read_csv('data/y_test.csv',header=0)
	testX1=testX.drop(['id'],axis=1)
	testY=testY['fault_severity']

	print("Random")
	random=RandomForestClassifier(n_estimators=2700)
	#best = 2500
	random.fit(X,y)
	print(random.score(X,y))
	print(random.score(testX1,testY))


	# prediction
	testy=random.predict_proba(testX1)

	pred_cols = ['predict_{}'.format(i) for i in range(3)]
	submission = pd.DataFrame(random.predict_proba(testX1),index=testX.id,columns=pred_cols)

	print(multiclass_log_loss(testY.values, submission.values))
	submission.to_csv('rf_output.csv', index_label='id')

def find_best_param():

	X= pd.read_csv('data/X_train.csv',header=0)
	y= pd.read_csv('data/y_train.csv',header=0)
	X= X.drop(['id'],axis=1)
	y=y['fault_severity']


	testX= pd.read_csv('data/X_test.csv',header=0)
	testY= pd.read_csv('data/y_test.csv',header=0)
	testX1=testX.drop(['id'],axis=1)
	testY=testY['fault_severity']

	ret=[]
	#est = list(range(100,3000,100))
	#est = list(range(2400,2850,50))
	est = list(range(2680,2720,5))
	#best with 2500
	for value in est:	
		rf = RandomForestClassifier(n_estimators = value,random_state=1)
		rf.fit(X,y)
		#print(rf.score(X,y))
		#print(rf.score(testX1,testY))

		# prediction
		#testy=rf.predict_proba(testX1)

		pred_cols = ['predict_{}'.format(i) for i in range(3)]
		submission = pd.DataFrame(rf.predict_proba(testX1),index=testX.id,columns=pred_cols)

		ret.append(multiclass_log_loss(testY.values, submission.values))
		print(value, multiclass_log_loss(testY.values, submission.values))


	plt.plot(est,ret)
	plt.ylabel('Multi Class Log Loss')
	plt.xlabel('Number Of Estimators')
	plt.show()

predict_rf()
