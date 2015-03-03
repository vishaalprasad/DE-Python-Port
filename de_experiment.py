import sys; sys.path = ['/raid/vprasad/pylearn2'] + sys.path

from sparserf_example import create_sparserf
from layer2_example import create_layer2
from dae_mlp import create_classifier

if __name__ == "__main__":
    create_sparserf(10, [[3, 0], [0, 3]], 'sparserf_example.pkl')
    create_layer2()
    create_classifier()
