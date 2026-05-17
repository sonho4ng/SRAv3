#!/bin/bash

echo 'Start run training scripts, logs are being saved to ${OUTPUT_DIR}/train.log'

# Hàm chạy 1 script với các tham số 100–500
run_block() {
    local name=$1
    local path=$2
    echo "===== Running ${name} ====="
    for seed in 100 200 300 400 500; do
        echo "Running ${name} - seed ${seed} ..."
        bash ${path} $seed
    done
}

## --- Classification ---
run_block "banking77.sh"              ./scripts/banking77.sh
run_block "imdb.sh"                   ./scripts/imdb.sh
run_block "patent.sh"                 ./scripts/patent.sh

## --- SentencePair ---
run_block "anli_r2.sh"                ./scripts/anli_r2.sh
run_block "control.sh"                ./scripts/control.sh
run_block "scitail.sh"                ./scripts/scitail.sh

## --- STS ---
run_block "SICK.sh"              ./scripts/SICK.sh
run_block "sts12.sh"             ./scripts/sts12.sh
run_block "stsb.sh"              ./scripts/stsb.sh


echo "=== All done ==="
