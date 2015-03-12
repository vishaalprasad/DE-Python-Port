def create_classifier(pkl_file, save_path):
    # Create the dataset
    from de.datasets import Sergent
    from de import sparserf_autoencoder
    Sergent.create_datasets(overwrite=True)

    # Train the network.
    #from pylearn2.scripts.train import train

    layer1_yaml = open('dae_mlp.yaml', 'r').read()
    hyper_params = {'pkl_file' : pkl_file,
                    'save_path' : save_path}
    layer1_yaml = layer1_yaml % hyper_params
    from pylearn2.config import yaml_parse
    train = yaml_parse.load(layer1_yaml)
    train.main_loop()
    #train(config=layer1_yaml)

if __name__ == "__main__":
    create_classifier("sparserf_example.pkl", "final.pkl")
