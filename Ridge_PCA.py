import os
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, make_scorer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

np.random.seed(7)

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
    f_value = fscore(y_val, pred)
    mae_value = maeOverFscore(y_val, pred)
    
    print(f"fscore : {f_value} \t maeOverFscore : {mae_value}")
    
    return f_value, mae_value

base = os.getcwd()
data_path = os.path.join(base, 'data')
submit_path = os.path.join(base, 'submit')

def load_data(name):
    return np.load(os.path.join(data_path, f"{name}.npy"))

def load(name):
    if name == "test" :
        return load_data('x', 'test')
    return (load_data(f'x_{name}'), load_data(f'y_{name}'))

def reshape(data):
    return data.reshape(data.shape[0] * 40 * 40, data.shape[-1])

data = load_data('EDA')

selectK_8 = [2, 3, 4, 5, 6, 7, 11, 13]

X = data[:, selectK_8]
Y = data[:,  -1]

del data
print(X.shape, Y.shape)

kfold = KFold(n_splits=4, random_state=7, shuffle=True)

for (train_idx, val_idx) in kfold.split(Y):
    clf = Ridge(alpha=10.0)
    sclaer= PCA(n_components=1, random_state=7)
    
    x = scaler.fit_transform(X[train_idx, :])

    ridge = clf.fit(x, Y[train_idx])

    del x
    x = scaler.transform(X[val_idx, :])

    pred = clf.predict(x)
    score(Y[val_idx], pred)

    del x

clf = Ridge(alpha=10.0)
sclaer= PCA(n_components=1, random_state=7)
X = scaler.fit_transform(X)
ridge = clf.fit(X, Y)

x_test = load('test')
x_test = reshape(x_test)
x_test = scaler.transform(x_test)

pred = clf.predict(x_test)

submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
submission.iloc[:, 1:] = pred.reshape(-1, 1600)

submission.to_csv(os.path.join(submit_path, 'Ridge_PCA_n_1.csv'), index = False)
