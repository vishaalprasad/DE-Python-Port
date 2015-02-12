
if __name__ == "__main__":

    # Create the dataset
    from de.datasets import VanHateren
    VanHateren.create_datasets()

    # Train the network.
    from pylearn2.scripts.train import train
    train(config="sparserf.yaml")

    # Visualize the weights
    from pylearn2.scripts.show_weights import show_weights
    show_weights(model_path="sparserf.pkl", border=True)

    # Visualize the reconstruction
    from de.compare_reconstruct import compare_reconstruction
    compare_reconstruction(model_path="sparserf.pkl")
