import sys
import os

from pylearn2.utils import serial
from pylearn2.config import yaml_parse

from de.datasets import Sergent

try:
    model_path = sys.argv[1]
except IndexError:
    print "Usage: predict.py <model file>"
    quit()

try:
    model = serial.load( model_path )
except Exception, e:
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e




X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)

from theano import tensor as T

y = T.argmax(Y, axis=1)

from theano import function

f = function([X], y)

imgs = serial.load("sergent.pkl")
img = imgs[0]
import numpy as np
import pdb; pdb.set_trace()
y = f(np.resize(img, (32, 32)))

