import tensorflow as tf
import numpy as np 
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.eager import context
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import tensorflow_probability as tfp

def stochastic_activity(x,seed=None,name=None):
    ops.convert_to_tensor(x,name='x')
    if not x.dtype.is_floating:
        raise ValueError("x has to be a floating point tensor since it's going to"
                         " be scaled. Got a {} tensor instead.".format(x.dtype))

    # random_ratios = random_ops.random_uniform(array_ops.shape(x),dtype=x.dtype)
    uniform = tfp.distributions.Uniform(low=0.0,high=1.0)
    random_ratios = uniform.sample(array_ops.shape(x))
    ret = x * random_ratios * 2.0
    if not context.executing_eagerly():
        ret.set_shape(x.get_shape())
    return ret

def gaussian_stochastic_activity(x,seed=None,name=None):
    ops.convert_to_tensor(x,name='x')
    if not x.dtype.is_floating:
        raise ValueError("x has to be a floating point tensor since it's going to"
                         " be scaled. Got a {} tensor instead.".format(x.dtype))

    random_ratios = random_ops.random_normal(array_ops.shape(x),mean=1.0,stddev=x/np.sqrt(25),dtype=x.dtype)
    ret = x * random_ratios 
    if not context.executing_eagerly():
        ret.set_shape(x.get_shape())
    return ret

def exponential_stochastic_activity(x,seed=None,name=None):
    ops.convert_to_tensor(x,name='x')
    if not x.dtype.is_floating:
        raise ValueError("x has to be a floating point tensor since it's going to"
                         " be scaled. Got a {} tensor instead.".format(x.dtype))

    random_ratios = random_ops.random_gamma(array_ops.shape(x),1.0,beta=1.0,dtype=x.dtype)
    ret = x * random_ratios 
    if not context.executing_eagerly():
        ret.set_shape(x.get_shape())
    return ret
    
class StochasticActivity(tf.keras.layers.Layer):
    def __init__(self, seed=None, **kwargs):
        super(StochasticActivity, self).__init__(**kwargs)
        self.seed = seed
        self.supports_masking = True
    def call(self,inputs,training=None):
        def dropped_inputs():
            return stochastic_activity(inputs,seed=self.seed)
            # return gaussian_stochastic_activity(inputs,seed=self.seed)
            # return exponential_stochastic_activity(inputs,seed=self.seed)
        ret = tf.keras.backend.in_train_phase(dropped_inputs,inputs,training=training)
        ret.set_shape(inputs.get_shape())
        return ret

    def get_config(self):
        config = {'seed':self.seed}
        base_config = super(StochasticActivity,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,input_shape):
        return input_shape
