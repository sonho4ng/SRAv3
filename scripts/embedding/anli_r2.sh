#! /bin/bash

SEED=$1

# ==== Định nghĩa các biến ====
BASE_PATH=.
DATA_NAME="anli_r2"
OUTPUT_DIR="${BASE_PATH}/outputs/${DATA_NAME}/seed-${SEED}"
CKPT_NAME="bert-llm2vec-${DATA_NAME}-checkpoint"

mkdir -p ${OUTPUT_DIR}

# Gom tham số vào OPTS
OPTS=""
# data
OPTS+=" --train_data ${BASE_PATH}/data/${DATA_NAME}/train.csv"
OPTS+=" --val_data ${BASE_PATH}/data/${DATA_NAME}/val.csv"
OPTS+=" --test_data ${BASE_PATH}/data/${DATA_NAME}/test.csv"

# training
OPTS+=" --num_train_epochs 10"
OPTS+=" --batch_size 8"
OPTS+=" --val_batch_size 32"
OPTS+=" --learning_rate 2e-5"
OPTS+=" --max_len 128"
OPTS+=" --num_labels 3"

# devices
OPTS+=" --teach_device auto"
OPTS+=" --student_device auto"

# loss
OPTS+=" --hard_label_loss_weight 0.5"
OPTS+=" --loss_type ce"
OPTS+=" --orthogonal True"
OPTS+=" --span_loss True"
OPTS+=" --der_loss True"
OPTS+=" --span_weight_pooling True"
OPTS+=" --span_loss_weight True"
OPTS+=" --p 1.0"

OPTS+=" --teacher_layers_mapping 23 24 25 26 27 28 29 30 31 32"
OPTS+=" --student_encoder_layers_finetuned 3 4 5 6 7 8 9 10 11 12"
OPTS+=" --hidden_loss_weights 1 1 1 1 1 1 1 1 1 1"


# models
OPTS+=" --teacher_embedding_dimension 4096"
OPTS+=" --output_dir ${OUTPUT_DIR}"
OPTS+=" --teacher_model McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
OPTS+=" --teacher_tokenizer McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
OPTS+=" --student_model google-bert/bert-base-uncased"
OPTS+=" --student_tokenizer google-bert/bert-base-uncased"

# hf token
OPTS+=" --hf_token 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"

# extra arguments
OPTS+=" --seed ${SEED}"
# OPTS+=" --teacher_sft None"
OPTS+=" --task_type sts"
OPTS+=" --teacher_mean_pooling True"

# ==== Gọi Python ====
python run_distill_llm2vec.py ${OPTS} >> ${OUTPUT_DIR}/train.log 2>&1
