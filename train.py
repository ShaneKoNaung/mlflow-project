from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import scipy

import mlflow

def train_linear_sklearn(model : linear_model ,X_train : scipy.sparse._csr.csr_matrix, 
                         y_train : scipy.sparse._csr.csr_matrix, 
                         X_val : scipy.sparse._csr.csr_matrix, 
                         y_val : scipy.sparse._csr.csr_matrix) -> linear_model:
    
    with mlflow.start_run() as run:

        lr = model()

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

    return lr


