import sys; sys.path = ['/raid/vprasad/pylearn2'] + sys.path

from pylearn2.scripts.show_weights import show_weights

from de.compare_reconstruct import compare_reconstruction


def create_sparserf(num_cons, sigma, weights_file, verbosity=1):
    import tempfile

    params = (num_cons, sigma, weights_file)

    # Create the dataset
    from de.datasets import VanHateren
    VanHateren.create_datasets()

    # Create the yaml file.
    _, config_fn = tempfile.mkstemp()
    with open("sparserf_template.yaml") as fp:
        # create yaml from templtes + params
        config_yaml = "".join(fp.readlines()) % params
    with open(config_fn, 'w') as config_fp:
        config_fp.write(config_yaml)

    # Train the network
    from pylearn2.scripts.train import train
    train(config=config_fn)

    # Visualize the weights
    if verbosity > 0:
        show_weights(model_path=weights_file, border=True)

    # Visualize the reconstruction
    if verbosity > 0:
        compare_reconstruction(model_path=weights_file)


if __name__ == "__main__":
    create_sparserf(10, [[3, 0], [0, 3]], 'sparserf_example.pkl')
