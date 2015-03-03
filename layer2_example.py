import sys;
sys.path = ['/raid/vprasad/pylearn2'] + sys.path
if __name__ == "__main__":
    import tempfile

    # Create the dataset
    from de.datasets import VanHateren
    VanHateren.create_datasets()

    #params = ()
    # Create the yaml file.
    _, config_fn = tempfile.mkstemp()
    with open("dae_l2.yaml") as fp:
        # create yaml from templtes + params
        config_yaml = "".join(fp.readlines())# % params
    with open(config_fn, 'w') as config_fp:
        config_fp.write(config_yaml)

    # Train the network
    from pylearn2.scripts.train import train
    train(config=config_fn)

    # Visualize the weights
    #from pylearn2.scripts.show_weights import show_weights
   # show_weights(model_path=weights_file, border=True)

    # Visualize the reconstruction
   # from de.compare_reconstruct import compare_reconstruction
   #compare_reconstruction(model_path=weights_file)
