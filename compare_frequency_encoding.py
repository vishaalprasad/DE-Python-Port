
if __name__ == "__main__":
    import os
    import sys
    import tempfile

    plotting_param = sys.argv[1] if len(sys.argv) > 1 else True

    hemi_params = {
        'left':  {'sigma': [[5, 0], [0, 5]]},
        'right': {'sigma': [[3, 0], [0, 3]]}, }

    # Create the dataset
    from de.datasets import VanHateren
    VanHateren.create_datasets()

    for hemi in hemi_params.keys():
        weights_file = '%s.pkl' % hemi
        if os.path.exists(weights_file):
            continue

        params = (  # must be in order...
            10,  # number of connections
            hemi_params[hemi]['sigma'],
            weights_file)  # weights file

        # Create the yaml file.
        _, config_fn = tempfile.mkstemp()
        with open("sparserf_template.yaml") as fp:
            # create yaml from templtes + params
            yaml_template = "".join(fp.readlines())
            config_yaml = yaml_template % params
        with open(config_fn, 'w') as config_fp:
            config_fp.write(config_yaml)

        # Train the network
        from pylearn2.scripts.train import train
        train(config=config_fn)

        # Visualize the weights
        # from pylearn2.scripts.show_weights import show_weights
        # show_weights(model_path=weights_file, border=True)

        # Visualize the reconstruction
        # from de.compare_reconstruct import compare_reconstruction
        # compare_reconstruction(model_path=weights_file)

    # Analyze frequency information
    from de.fft_analyze import hemisphericalDifferences
    hemisphericalDifferences('left.pkl', 'right.pkl', plotting=plotting_param)
