from pylearn2.config import yaml_parse

from de.datasets import Sergent
from de import sparserf_autoencoder


def create_classifier(autoencoder_path, save_path,
                      mlp_template='dae_mlp.yaml'):
    # Create the dataset
    Sergent.create_datasets(overwrite=True)

    # Train the network.
    hyper_params = {'pkl_file': autoencoder_path,
                    'save_path': save_path}
    with open(mlp_template, 'r') as fp:
        layer1_yaml = fp.read() % hyper_params

    train = yaml_parse.load(layer1_yaml)
    train.main_loop()


if __name__ == "__main__":
    create_classifier("sparserf_example.pkl", "final.pkl")
