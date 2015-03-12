import sys; sys.path = ['/raid/vprasad/pylearn2'] + sys.path

from sparserf_example import create_sparserf
from dae_mlp import create_classifier

if __name__ == "__main__":
    create_sparserf(10, [[4, 0], [0, 4]], 'left_hemisphere.pkl')
    create_classifier('left_hemisphere.pkl', 'left_final.pkl')

    create_sparserf(10, [[3, 0], [0, 3]], 'right_hemisphere.pkl')
    create_classifier('right_hemisphere.pkl', 'right_final.pkl')
