import sys
import os


import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
from theano import tensor as T

from de.datasets import Sergent


try:
    model_path = sys.argv[1]
except IndexError:
    print "Usage: predict.py <model file>"
    quit()

try:
    model = serial.load(model_path)
except Exception, e:
    print "Error while loading model path %s: %s" % (model_path, e)
    quit()

imgs = serial.load("sergent.pkl")

X = model.get_input_space().make_batch_theano(batch_size=imgs.shape[0])
Y = model.fprop(X)
Y = T.argmax(Y, axis=1)
f = function([X], Y)

y = f(imgs)
print "Computed targets:", y
