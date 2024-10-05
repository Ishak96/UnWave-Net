#!/bin/bash

args=(
    --config-name 'train.yaml'

    model=unwavenet.yaml

    datamodule.json_path="src/json/mayo.json"

    default_hp.num_detectors=100
    default_hp.num_cascades=16
    default_hp.add_noise=False
    default_hp.max_epochs=50

    default_hp.resume_from_checkpoint=null

    trainer.gradient_clip_val=1.0
    trainer.accelerator="cuda"
    trainer.devices=-1
    train=True
    test=False
    model.save_targets=False
)

# Replace defaults with user-provided args and retrieve --debug flag
debug=0
for arg in "$@"; do
    if [[ "$arg" == *"="* ]]; then
        args+=("$arg")
    elif [ "$arg" == "--debug" ]; then
        debug=1
    fi
done

# Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
export HYDRA_FULL_ERROR=1

# Set the environment variable CUDA_VISIBLE_DEVICES to control which GPU is used.
export CUDA_VISIBLE_DEVICES=0

# run the app
if [ $debug -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5678 --wait-for-client my_app.py "${args[@]}"
else
    python app.py "${args[@]}"
fi
