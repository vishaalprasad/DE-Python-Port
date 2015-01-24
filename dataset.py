from __future__ import print_function

from pylearn2.utils import serial
from pylearn2.utils import string_utils

from vanhateren import VANHATEREN


DATA_DIR = string_utils.preprocess('${PYLEARN2_DATA_PATH}/vanhateren')
ALL_DATASETS = ['train', 'test', 'valid']

def create_datasets(datasets=ALL_DATASETS, output_dir=DATA_DIR):
	serial.mkdir( output_dir )

	for dataset_name in list(datasets):
	    print("Loading the %s data" % dataset_name)
	    dataset = VANHATEREN(which_set=dataset_name)

	    print("Saving the test data")
	    dataset.use_design_loc(output_dir + '/%s.npy' % dataset_name)
	    serial.save(output_dir + '/%s.pkl' % dataset_name, dataset)


if __name__ == "__main__":
	create_datasets()