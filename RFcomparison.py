from sklearn.ensemble import RandomForestRegressor
from scaler import train_scaler,scale_data,inverse_scale_data,inverse_scale_uncertainties
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score,mean_squared_error

def random_forest(x_train=None,x_test=None,y_train=None,y_test=None,number_trees=100):
    try:
        _ = x_train.shape, x_test.shape, y_train.shape, y_test.shape
    except AttributeError:
        print('Data do not seem to be numpy arrays or tensors?')
        exit()

    y_train_trans = np.transpose([y_train])
    y_test_trans  = np.transpose([y_test])

    x_scalers,x_train = train_scaler(x_train)
    x_test = scale_data(x_test,x_scalers)
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')
    joblib.dump(x_scalers,'saved_models/x_scalers.pkl')

    y_scalers,y_train = train_scaler(y_train_trans)
    y_test = scale_data(y_test_trans,y_scalers)
    joblib.dump(y_scalers,'saved_models/y_scalers.pkl')

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    model = RandomForestRegressor(n_estimators=number_trees,n_jobs=-1)
    model.fit(x_train,y_train)

    predictions = np.array([est.predict(x_test) for est in model.estimators_])

    predictions_all = np.mean(predictions,axis=0)
    uncertainties_all = np.std(predictions,axis=0)

    predictions_all = np.transpose([predictions_all])
    uncertainties_all = np.transpose([uncertainties_all])

    predictions_all = inverse_scale_data(predictions_all,y_scalers)
    uncertainties_all = inverse_scale_uncertainties(uncertainties_all,y_scalers)

    print('Accuracy all: R^2={:.4f}'.format(r2_score(y_test_trans,predictions_all)))
    print('RMSE all: ${:.4f}'.format(1000*np.sqrt(mean_squared_error(y_test_trans,predictions_all))))

    predictions_all = np.concatenate([y_test_trans,predictions_all,uncertainties_all],axis=-1)

    if not os.path.isdir('boston_housing/data'):
        os.makedirs('boston_housing/data')
    pd.DataFrame(predictions_all).to_csv('boston_housing/data/boston_RF_predictions.csv',header=None,index=None)



    