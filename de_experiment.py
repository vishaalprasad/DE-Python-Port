import numpy as np
import os
import sys

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
from theano import tensor as T

from de.datasets import Sergent


def collect_results(model_path):
    try:
        print("Loading model from %s..." % model_path)
        model = serial.load(model_path)
    except Exception, e:
        print "Error while loading model path %s: %s" % (model_path, e)
        raise

    data_path = os.path.join(Sergent.DATA_DIR, "train.pkl")
    print("Loading Sergent dataset from %s..." % data_path)
    dataset = serial.load(data_path)

    X = model.get_input_space().make_batch_theano(batch_size=dataset.X.shape[0])
    Y = model.fprop(X)
    Y = T.argmax(Y, axis=1)
    f = function([X], Y)

    y = f(dataset.X)

    return model, y


def show_results(left_mlp_pkl, right_mlp_pkl):

    # Collect the results
    models = dict()
    ys = dict()
    for model_path, model_name in ((left_mlp_pkl, 'left'),
                                   (right_mlp_pkl, 'right')):
        models[model_name], ys[model_name] = collect_results(model_path)
        print "Computed targets:", ys[model_name]

    # Plot the results
    print "Plotting NYI"


if __name__ == "__main__":
    from sparserf_example import create_sparserf
    from classifier import create_classifier

    create_sparserf(num_cons=10, sigma=[[4, 0], [0, 4]],
                    weights_file='left_hemisphere.pkl', verbosity=0)
    create_classifier('left_hemisphere.pkl', 'left_final.pkl')

    create_sparserf(num_cons=10, sigma=[[2, 0], [0, 2]],
                    weights_file='right_hemisphere.pkl', verbosity=0)
    create_classifier('right_hemisphere.pkl', 'right_final.pkl')

    show_results('left_final.pkl', 'right_final.pkl')
