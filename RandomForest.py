import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, make_scorer

def mae(y_true, y_pred) :
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    y_pred = y_pred.reshape(1, -1)[0]
    over_threshold = y_true >= 0.1
    
    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))

def fscore(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    y_pred = y_pred.reshape(1, -1)[0]
    remove_NAs = y_true >= 0
    
    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)
    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)
    
    return (f1_score(y_true, y_pred))

def maeOverFscore(y_true, y_pred):
    return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)

def score(y_val, pred):
    print(f"fscore : {fscore(y_val, pred)} \t maeOverFscore : {maeOverFscore(y_val, pred)}")

np.random.seed(7)
selectK = [2, 3, 4, 5, 6, 7, 11, 13]

print("data load")
data = np.load(os.path.join(os.getcwd(), 'data', "EDA.npy"))

print("data seperate")
X = data[:, selectK]
Y = data[:,  -1]

del data

for estimator in [100, 400]:
	print("train")
	clf = RandomForestRegressor(n_estimators=estimator, random_state=7).fit(X, Y)

	print("test load")
	x_test = np.load(os.path.join(os.getcwd(), 'data', "test.npy"))
	x_test = x_test.reshape(x_test.shape[0] * 40 * 40, -1)
	x_test = x_test[:, selectK]

	print("predict")
	pred = clf.predict(x_test)

	print("submit load")
	submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
	submission.iloc[:, 1:] = pred.reshape(-1, 1600)

	print("submit save")
	path = os.path.join(submit_path, 'RandomForest_{}_selectK8.csv'.format(estimator))
	submission.to_csv(path, index = False)
