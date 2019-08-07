import numpy as np 
import tensorflow as tf
from callbacks import step_decay, momentum_schedule, MomentumScheduler, IncorrectHistory
from stochastic_activities import StochasticActivity,stochastic_activity
import os

def dense_net(x_train=None,x_test=None,y_train=None,y_test=None,num_classes=10,dropout=False,stochastic=False,stochact=False,epochs=5):
    try:
        _ = x_train.shape, x_test.shape, y_train.shape, y_test
    except AttributeError:
        print('Data do not seem to be numpy arrays or tensors?')
        exit()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    if dropout and stochastic:
        print('Stochastic units do not work with dropout yet')
        exit()

    if dropout:
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ])
    elif stochastic:
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        StochasticActivity(),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        StochasticActivity(),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        StochasticActivity(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ])
    else:
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ])

    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    moment = MomentumScheduler(momentum_schedule)

    sgd = tf.keras.optimizers.SGD(clipnorm=15,momentum=momentum_schedule(0),lr=step_decay(0))
    model.compile(optimizer=sgd,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = IncorrectHistory()
    model.fit(x_train, y_train, epochs=epochs, batch_size=100, callbacks=[lrate,moment,history], 
              validation_data=(x_test, y_test))
    model.save(os.path.join('saved_models','dense_net.h5'))
    incorrect_history = np.rint(len(y_test)*np.array(history.incorrect))
    np.savetxt('incorrect_history.csv',incorrect_history,delimiter=',')
    predictions = model.predict(x_test)
    correct = [True if np.argmax(pred)==true else False for pred,true in zip(predictions,y_test)]
    print('Number incorrect: {}'.format(len(correct)-np.sum(correct)))
    print('Accuracy on test: {:.3f}'.format(np.mean(correct)))