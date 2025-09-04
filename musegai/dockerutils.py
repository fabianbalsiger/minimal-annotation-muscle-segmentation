import pathlib
import shutil
import docker
from . import docker_template

"""
TODO:
- create one image per model
- use consistent tags (or simply use `:latest`)
"""

# repository
TRAIN_IMAGE = "trainer-museg:latest"


def list_models(local=True):
    """list existing models"""
    models = []
    if local:
        client = docker.from_env()
        for image in client.images.list():
            try:
                if image.tags[0].startswith("museg"):
                    models.append(image.tags[0])
            except IndexError:
                pass
    return models


def get_model_info(model):
    """return info of the designated model"""
    client = docker.from_env()
    for image in client.images.list():
        try:
            if model == image.tags[0]:
                return image.labels
        except IndexError:
            pass
    raise ValueError(f'Unknown model: {model}')


def run_inference(model, dirname):
    """Run inference"""
    client = docker.from_env()
    image = _get_image(model)
    _pull_image(image)
    print(f"Running model '{model}' (`{image}`)")
    client.containers.run(
        image,
        remove=True,
        ipc_mode="host",
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        volumes={dirname: {"bind": "/data", "mode": "rw"}},
    )

    # copy labels.txt into bound directory outdir
    client.containers.run(
        model,
        ["cp", "/labels.txt", "/data/labels.txt"],
        entrypoint="",
        remove=True,
        volumes={dirname: {"bind": "/data", "mode": "rw"}},
    )


def check_training():
    """check if training image available"""
    # TODO
    return True


def run_training(model, dirname, folds=(0, 1, 2, 3, 4), nepoch=1000, preprocess=True, continue_training=False):
    """Run training"""
    dirname = str(pathlib.Path(dirname).resolve())
    client = docker.from_env()
    image = TRAIN_IMAGE
    try:
        _pull_image(image)
    except:
        raise RuntimeError('Could not find museg-ai training image')

    print(f"Training model '{model}'")

    def run(entrypoint, cmd):
        container = client.containers.run(
            image,
            cmd,
            entrypoint=entrypoint,
            remove=True,
            detach=True,
            ipc_mode="host",
            device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
            volumes={dirname: {"bind": "/nnunet", "mode": "rw"}},
        )
        # print logs
        for line in container.logs(stream=True, follow=True):
            print(line.decode("utf-8").strip())
        return container.status

    # preprocess
    if preprocess:
        status = run("nnUNetv2_plan_and_preprocess", ["-d", "001", "-c", "3d_fullres", "--verify_dataset_integrity"])
        # "-pl nnUNetPlannerResEncL" # (M/L/XL)

    opts = []
    if nepoch != 1000:
        opts.extend(["-tr", f"nnUNetTrainer_{nepoch}epochs"])
    if continue_training:
        print("Warning: continue training")
        opts.extend(["--c"])

    # train
    # -p nnUNetResEncUNetLPlans # (M/L/XL)
    if 0 in folds:
        status = run("nnUNetv2_train", ["001", "3d_fullres", "0"] + opts)
    if 1 in folds:
        status = run("nnUNetv2_train", ["001", "3d_fullres", "1"] + opts)
    if 2 in folds:
        status = run("nnUNetv2_train", ["001", "3d_fullres", "2"] + opts)
    if 3 in folds:
        status = run("nnUNetv2_train", ["001", "3d_fullres", "3"] + opts)
    if 4 in folds:
        status = run("nnUNetv2_train", ["001", "3d_fullres", "4"] + opts)


def get_ressources():
    """get the path to the ressources folder"""
    here = pathlib.Path(__file__).parent
    return here / 'data'


def make_dockerfile(model, dirname, nchannel, folds=(0, 1, 2, 3, 4), nepoch=1000):
    """build inference docker"""
    client = docker.from_env()

    trainer = "nnUNetTrainer"
    if nepoch != 1000:
        trainer = f"nnUNetTrainer_{nepoch}epochs"

    # get docker template
    dockerfile = docker_template.make_docker(model, dirname, folds, nchannel=nchannel, trainer=trainer)

    # get the requirement file
    ressources_dir = get_ressources()
    shutil.copy((ressources_dir / "requirements.txt"), dirname)

    # dockerfile writing
    with open(dirname / "Dockerfile", "w") as fp:
        fp.write(dockerfile)

    # build image
    print("Run the following command to build the model's docker:")
    print(f"\tdocker build {dirname}/ --tag {model}")
    # image, logs = client.images.build(path=str(dirname), quiet=False, forcerm=True, rm=True)
    # for chunk in logs:
    #     if not "stream" in chunk:
    #         continue
    #     for line in chunk["stream"].splitlines():
    #         print(line)


# private


def _get_image(model):
    """Get Docker image name."""
    if ":" in model:  # if arg is a valid docker name
        return model
    # TODO
    # return default model if input is not a valid docker name
    else:
        return f"fabianbalsiger/museg:{model}"


def _pull_image(image):
    """Pull a Docker image if not exists."""
    client = docker.from_env()
    if not client.images.list(name=image):
        print(f"Pulling image `{image}`, this may take a while...")
        client.images.pull(image)
