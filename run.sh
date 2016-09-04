#!/bin/bash -l

echo 'Running Model'


# change this to the repo path!
python3  ./train.py \
        --model "DAModel" \
        --data_path "./snli_1.0" \
        --weights_dir "./out/test_run" \
        --verbose True \
        --debug  \
