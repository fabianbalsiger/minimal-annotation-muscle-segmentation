"""API for muscle segmentation."""

from . import dockerutils, train, infer, test, io

is_image = io.is_image
get_model_info = dockerutils.get_model_info
list_models = dockerutils.list_models
run_model = infer.infer
train_model = train.train
dockerize_model = train.dockerize
test_model = test.test
compare_preds = test.compare
