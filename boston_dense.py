import numpy as np 
import tensorflow as tf
from callbacks import step_decay, momentum_schedule, MomentumScheduler, IncorrectRMSEHistory
from stochastic_activities import StochasticActivity,stochastic_activity
import os
import pandas as pd
from scaler import train_scaler,scale_data,inverse_scale_data,inverse_scale_uncertainties
from sklearn.externals import joblib
from sklearn.metrics import r2_score,mean_squared_error


def predict_with_uncertainty(f, x, output_length, n_iter=100):
    result = np.zeros((n_iter,) + (x.shape[0],output_length))

    for i in range(n_iter):
        result[i,:,:] = f((x,1))[0]

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction,uncertainty

def boston_net(x_train=None,x_test=None,y_train=None,y_test=None,dropout=False,stochastic=False,stochact=False,epochs=5):
    try:
        _ = x_train.shape, x_test.shape, y_train.shape, y_test.shape
    except AttributeError:
        print('Data do not seem to be numpy arrays or tensors?')
        exit()

    y_train_trans = np.transpose([y_train])
    y_test_trans  = np.transpose([y_test])

    x_scalers,x_train = train_scaler(x_train)
    x_test = scale_data(x_test,x_scalers)
    joblib.dump(x_scalers,'saved_models/x_scalers.pkl')

    y_scalers,y_train = train_scaler(y_train_trans)
    y_test = scale_data(y_test_trans,y_scalers)
    joblib.dump(y_scalers,'saved_models/y_scalers.pkl')

    if dropout and stochastic:
        print('Stochastic units do not work with dropout yet')
        exit()   

    input_ = tf.keras.layers.Input(shape=(13,))
    
    if dropout:
        encoding = tf.keras.layers.Dense(800,activation='relu')(input_)
        encoding = tf.keras.layers.Dropout(0.5)(encoding)
        encoding = tf.keras.layers.Dense(800,activation='relu')(encoding)
        encoding = tf.keras.layers.Dropout(0.5)(encoding)
        output = tf.keras.layers.Dense(1,activation='linear')(encoding)
    elif stochastic:
        encoding = tf.keras.layers.Dense(800,activation='relu')(input_)
        encoding = StochasticActivity()(encoding)
        encoding = tf.keras.layers.Dense(800,activation='relu')(encoding)
        encoding = StochasticActivity()(encoding)
        output = tf.keras.layers.Dense(1,activation='linear')(encoding)
    else:
        encoding = tf.keras.layers.Dense(800,activation='relu')(input_)
        encoding = tf.keras.layers.Dense(800,activation='relu')(encoding)
        output = tf.keras.layers.Dense(1,activation='linear')(encoding)

    model = tf.keras.models.Model(inputs=input_,outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')
    

    model.fit(x_train,y_train, epochs=epochs, batch_size=16, 
              validation_data=(x_test, y_test))
    model.save(os.path.join('saved_models','boston_housing_net.h5'))

    predictions_all = model.predict(x_test)

    func = tf.keras.backend.function([model.layers[0].input, tf.keras.backend.learning_phase()],[model.layers[-1].output])

    predictions_100, uncertainties_100 = predict_with_uncertainty(func,x_test,1,n_iter=100)
    _, uncertainties_99 = predict_with_uncertainty(func,x_test,1,n_iter=99)





    predictions_all = inverse_scale_data(predictions_all,y_scalers)
    predictions_100 = inverse_scale_data(predictions_100,y_scalers)
    uncertainties_100 = inverse_scale_uncertainties(uncertainties_100,y_scalers)
    uncertainties_99 = inverse_scale_uncertainties(uncertainties_99,y_scalers)

    print('Accuracy all: R^2={:.4f}'.format(r2_score(y_test_trans,predictions_all)))
    print('Accuracy 100: R^2={:.4f}'.format(r2_score(y_test_trans,predictions_100)))

    print('RMSE all: ${:.4f}'.format(1000*np.sqrt(mean_squared_error(y_test_trans,predictions_all))))
    print('RMSE 100: ${:.4f}'.format(1000*np.sqrt(mean_squared_error(y_test_trans,predictions_100))))

    predictions_all = np.concatenate([y_test_trans,predictions_all,uncertainties_99],axis=-1)
    predictions_100 = np.concatenate([y_test_trans,predictions_100,uncertainties_100],axis=-1)

    pd.DataFrame(predictions_all).to_csv('boston_housing/data/boston_predictions_all.csv',header=None,index=None)
    pd.DataFrame(predictions_100).to_csv('boston_housing/data/boston_predictions_100.csv',header=None,index=None)