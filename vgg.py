import tensorflow as tf
import numpy as np
from stochastic_activities import StochasticActivity
from callbacks import resnet_lr_schedule,resnetv2_lr_schedule,TestAccuracy,momentum_schedule,MomentumScheduler
import os
import matplotlib.pyplot as plt

def layers_output(model,layers,x):
    
    func = tf.keras.backend.function([model.layers[0].input, tf.keras.backend.learning_phase()],[model.layers[el].output for el in layers])

    outputs = func([x,1])

    return outputs

def save_bottlebeck_features(x_train,x_test):
    print('Now saving bottleneck features')
    input_shape = x_train.shape[1:]
    input_tensor = tf.keras.layers.Input(shape=input_shape)

    # build the VGG16 network
    model = tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet',input_tensor=input_tensor)

    
    bottleneck_features_train = model.predict(x_train)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    bottleneck_features_validation = model.predict(x_test)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def vgg(x_train=None,x_test=None,y_train=None,y_test=None,num_classes=10,batch_size=1000,epochs=5,subtract_pixel_mean=True):
    #Normalise the data
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    #If subtracting the pixel mean
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train,axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    
    #Save the bottleneck features
    if not os.path.isfile('bottleneck_features_train.npy'):
        save_bottlebeck_features(x_train,x_test)

    #Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train,num_classes=num_classes)
    y_test  = tf.keras.utils.to_categorical(y_test,num_classes=num_classes)

    #Read in bottlenecked data
    x_train = np.load(open('bottleneck_features_train.npy','rb'))
    x_test  = np.load(open('bottleneck_features_validation.npy','rb'))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    # model.add(StochasticActivity())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(5e-4)))
    # model.add(StochasticActivity())
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(5e-4)))
    # model.add(StochasticActivity())
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(5e-4)))
    # model.add(StochasticActivity())
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

    #Prepare save structure
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'resnet.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir,model_name)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                    monitor='acc',
                                                    save_best_only=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    #Run training
    history = TestAccuracy()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[checkpoint,history],
              validation_data=(x_test, y_test))

    activations1 = layers_output(model,[-4],x_test)[0]
    activations2 = layers_output(model,[-3],x_test)[0]
    activations3 = layers_output(model,[-2],x_test)[0]
    np.savetxt('activations.csv',np.concatenate([activations1,activations2,activations3],axis=1),delimiter=',')

    #Generate predictions
    predictions = model.predict(x_test)
    incorrect_history = np.array(history.incorrect)
    np.savetxt('test_accuracy.csv',incorrect_history,delimiter=',')
    if len(y_test.shape)==2:
        correct = [True if np.argmax(pred)==np.argmax(true) else False for pred,true in zip(predictions,y_test)]
    elif len(y_test.shape)==1:
        correct = [True if np.argmax(pred)==true else False for pred,true in zip(predictions,y_test)]
    print('Number incorrect: {}'.format(len(correct)-np.sum(correct)))
    print('Accuracy on test: {:.3f}'.format(np.mean(correct)))