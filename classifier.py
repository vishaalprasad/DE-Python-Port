"""
This file constructs and trains the classifier network
"""
import os

from pylearn2.config import yaml_parse

from de.datasets import Sergent
from de import sparserf_autoencoder


def create_classifier(autoencoder_path, save_path,
                      mlp_template='classifier.yaml',
                      overwrite=False):
    # Create the dataset
    Sergent.create_datasets(overwrite=overwrite)

    # Train the network.
    if overwrite or not os.path.exists(save_path):
        hyper_params = {'autoencoder_path': autoencoder_path,
                        'save_path': save_path}
        with open(mlp_template, 'r') as fp:
            layer1_yaml = fp.read() % hyper_params

        print("Constructing the classifier model...")
        train = yaml_parse.load(layer1_yaml)

        print("Training the classifier model...")
        train.main_loop()


if __name__ == "__main__":
    create_classifier("sparserf_example.pkl", "final.pkl")
