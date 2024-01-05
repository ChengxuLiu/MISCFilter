from data.dataset_RGB import *


def get_training_data(rgb_dir, meta, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderFileTrain(rgb_dir, meta, img_options)

def get_validation_data(rgb_dir, meta, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderFileVal(rgb_dir, meta, img_options)

def get_test_data(meta, input_dir, target_dir, img_options):
    assert os.path.exists(meta)
    assert os.path.exists(input_dir)
    assert os.path.exists(target_dir)
    return DataLoaderFileTest(meta, input_dir, target_dir, img_options)
