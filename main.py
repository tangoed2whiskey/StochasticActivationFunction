import numpy as np 
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

def main(dataset=None,method=None,**kwargs):
    if dataset=='mnist' or dataset=='MNIST':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        num_classes=10
        if method=='resnet':
            x_train = x_train.reshape(x_train.shape+(1,))
            x_test = x_test.reshape(x_test.shape+(1,))
    elif dataset=='cifar' or dataset=='CIFAR':
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train),(x_test, y_test) = cifar.load_data()
        num_classes=10
    elif dataset=='boston':
        from sklearn.datasets import load_boston
        boston_dataset = load_boston()
        boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
        boston['MEDV'] = boston_dataset.target
        x_train,x_test,y_train,y_test = train_test_split(boston.loc[:,boston.columns!='MEDV'],boston['MEDV'],test_size=0.2,random_state=5)
        x_train,x_test,y_train,y_test = x_train.values,x_test.values,y_train.values,y_test.values
    else:
        print('Unknown dataset {}, please give a known dataset'.format(dataset))
        exit()

    if method=='dense':
        from dense_net import dense_net
        these_kwargs = {key:kwargs[key] for key in ['stochastic','dropout','epochs'] if key in kwargs}
        dense_net(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,num_classes=num_classes,**these_kwargs)
    elif method=='resnetv2':
        from vgg import vgg
        these_kwargs = {key:kwargs[key] for key in ['epochs','batch_size','subtract_pixel_mean'] if key in kwargs}
        vgg(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,num_classes=num_classes,**these_kwargs)
    elif method=='dense_boston':
        from boston_dense import boston_net
        these_kwargs = {key:kwargs[key] for key in ['stochastic','dropout','epochs'] if key in kwargs}
        boston_net(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,**these_kwargs)
    elif method=='RF':
        from RFcomparison import random_forest
        these_kwargs = {key:kwargs[key] for key in ['number_trees'] if key in kwargs}
        random_forest(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,**these_kwargs)
    else:
        print('Unknown method {}, please give a known method'.format(method))
        exit()




if __name__=='__main__':
    '''
    Pick one of the below operations
    '''
    # main(dataset='mnist',method='dense',stochastic=True,dropout=False,epochs=2000)
    # main(dataset='cifar',method='vgg',stochastic=False,dropout=False,epochs=250)
    # main(dataset='boston',method='dense_boston',stochastic=True,dropout=False,epochs=100)
    # main(dataset='boston',method='RF',number_trees=100)