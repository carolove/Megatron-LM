#!/bin/bash

# Runs the "7B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/workspace/checkpoints/llama-2-7b-mcore
TOKENIZER_MODEL=/workspace/model/llama-2-7b/tokenizer.model
TENSORBOARD_LOGS_PATH=/workspace/tensorboard_logs
DATA_PATH=/workspace/data/my-gpt2_text_document


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

MODEL_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --ffn-hidden-size 11008 \
    --position-embedding-type rope \
    --max-position-embeddings 4096 \
    --seq-length 4096 
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 949,50,1
"

TRAINING_ARGS="
    --use-legacy-models \
    --ckpt-format torch \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-iters 3000 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --bf16 \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 30 \
    --no-load-rng \
    --no-load-optim \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --use-flash-attn \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32
"

MODEL_PARALLEL_ARGS="
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 
"

EVAL_AND_LOGGING_ARGS="
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --eval-iters 10 \
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
" 

torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${MODEL_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    ${EVAL_AND_LOGGING_ARGS}
