# train_model.py
import pathlib
import sys
import joblib
import mlflow

import pandas as pd
from hyperopt import hp
from sklearn.model_selection import train_test_split
from hyperopt.pyll.base import scope
from sklearn.metrics import f1_score, precision_score, accuracy_score, roc_auc_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBRegressor


def find_best_model(xtrain,ytrain,xtest,ytest):

    # defining space 
    space ={
        #'penalty':hp.choice('penalty',['l1','l2','elasticnet']),
        'C':hp.uniform('C',1,3),
        #'solver':hp.choice('solver',['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
    }

    # function to train the model and evaluate the best one
    def objective(params):
        #penalty = params['penalty']
        C = params['C']
        #solver = params['solver']

        with mlflow.start_run():

            lr = LogisticRegression(C=C) #penalty=penalty,,solver=solver
            model=lr.fit(xtrain,ytrain)

            pred = model.predict(xtest)

            f_score = f1_score(ytest,pred)
            auc_roc = roc_auc_score(ytest,pred)

            #mlflow.log_param('penalty',penalty)
            mlflow.log_param('C',C)
            #mlflow.log_param('solver',solver)

            mlflow.log_metric('f1_score',f_score)
            mlflow.log_metric('auc_roc',auc_roc)

        return auc_roc            
    
    # findind the best model
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                trials=Trials(),
                max_evals=10)
    
    # logging the model
    with mlflow.start_run():
        mlflow.log_params(best)

        # train the model with best parameters
        final_model = LogisticRegression(C=best['C'],random_state= 42) #penalty=best['penalty']solver=best['solver'],
        final_model.fit(xtrain,ytrain)

        mlflow.sklearn.log_model(final_model,"Best_Model")

    return final_model


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/model.joblib")


def main():

    curr_dir = pathlib.Path(__file__)

    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file

    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True , exist_ok= True)

    TARGET = 'Class'
    train_features = pd.read_csv(data_path + '/train.csv')
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    trained_model = find_best_model(X_train, y_train, X_test, y_test)

    save_model(trained_model, output_path)

if __name__ == '__main__':
    main()






