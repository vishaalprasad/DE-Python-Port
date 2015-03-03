# Create the dataset
import sys
sys.path = ['/raid/vprasad/pylearn2'] + sys.path

from de.datasets import Sergent
from de import sparserf_autoencoder
Sergent.create_datasets(overwrite=True)

# Train the network.
from pylearn2.scripts.train import train
train(config="dae_mlp.yaml")

