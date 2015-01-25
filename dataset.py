from __future__ import print_function

import numpy as np
import os

from pylearn2.utils import serial
from pylearn2.utils import string_utils

from vanhateren import VANHATEREN


DATA_DIR = string_utils.preprocess('${PYLEARN2_DATA_PATH}/vanhateren')
ALL_DATASETS = ['train', 'test', 'valid']

def create_datasets(datasets=ALL_DATASETS, output_dir=DATA_DIR, overwrite=False):
    serial.mkdir(output_dir)

    for dataset_name in list(datasets):
        file_path_fn = lambda ext: os.path.join(output_dir, '%s.%s' % (dataset_name, ext))

        output_files = dict([(ext, file_path_fn(ext)) for ext in ['pkl', 'npy']])
        files_missing = np.any([not os.path.isfile(f) for f in output_files.values()])

        if overwrite or np.any(files_missing):
            print("Loading the %s data" % dataset_name)
            dataset = VANHATEREN(which_set=dataset_name)

            print("Saving the %d data" % dataset_name)
            dataset.use_design_loc(output_files['npy'])
            serial.save(output_files['pkl'], dataset)


if __name__ == "__main__":
    create_datasets(overwrite=True)
