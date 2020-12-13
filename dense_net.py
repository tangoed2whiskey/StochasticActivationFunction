import numpy as np 
import tensorflow as tf
from callbacks import step_decay, momentum_schedule, MomentumScheduler, IncorrectHistory
from stochastic_activities import StochasticActivity,stochastic_activity
import os

def layers_output(model,layers,x):
    
    func = tf.keras.backend.function([model.layers[0].input, tf.keras.backend.learning_phase()],[model.layers[el].output for el in layers])

    outputs = func([x,1])

    return outputs

def dense_net(x_train=None,x_test=None,y_train=None,y_test=None,num_classes=10,dropout=False,stochastic=False,stochact=False,epochs=5,**kwargs):
    tf.compat.v1.disable_eager_execution()
    try:
        _ = x_train.shape, x_test.shape, y_train.shape, y_test
    except AttributeError:
        print('Data do not seem to be numpy arrays or tensors?')
        exit()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    if dropout and stochastic:
        print('Stochastic units do not work with dropout yet')
        exit()
    try:
        if dropout:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
            tf.keras.layers.Dropout(kwargs['input_dropout']),
            tf.keras.layers.Dense(800, activation=tf.nn.relu),
            tf.keras.layers.Dropout(kwargs['hidden_dropout']),
            tf.keras.layers.Dense(800, activation=tf.nn.relu),
            tf.keras.layers.Dropout(kwargs['hidden_dropout']),
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
    except KeyError as e:
        print('Found a key error {}, please correct it'.format(e))
        exit()

    lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
    moment = MomentumScheduler(momentum_schedule)

    sgd = tf.keras.optimizers.SGD(clipnorm=15,momentum=momentum_schedule(0),lr=step_decay(0))
    model.compile(optimizer=sgd,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = IncorrectHistory()
    model.fit(x_train, y_train, epochs=epochs, batch_size=100, callbacks=[lrate,moment,history], 
              validation_data=(x_test, y_test))

    if dropout or stochastic:
        pre1 = -3
        pre2 = -5
    else:
        pre1 = -2
        pre2 = -3

    activations1 = np.mean(layers_output(model,[pre1],x_test)[0],axis=0)
    activations2 = np.mean(layers_output(model,[pre2],x_test)[0],axis=0)
    np.savetxt('activations.csv',np.concatenate([activations1,activations2]),delimiter=',')

    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')
    model.save(os.path.join('saved_models','dense_net.h5'))
    incorrect_history = np.rint(len(y_test)*np.array(history.incorrect))
    np.savetxt('incorrect_history.csv',incorrect_history,delimiter=',')
    predictions = model.predict(x_test)
    correct = [True if np.argmax(pred)==true else False for pred,true in zip(predictions,y_test)]
    print('Number incorrect: {}'.format(len(correct)-np.sum(correct)))
    print('Accuracy on test: {:.3f}'.format(np.mean(correct)))