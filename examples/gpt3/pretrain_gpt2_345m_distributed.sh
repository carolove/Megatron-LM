#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/workspace/model/megatron-models/345m-1dp-out
TENSORBOARD_LOGS_PATH=/workspace/tensorboard_logs
VOCAB_FILE=/workspace/model/gpt2-vocab/gpt2-vocab.json
MERGE_FILE=/workspace/model/gpt2-vocab/gpt2-merges.txt
DATA_PATH=/workspace/data/my-gpt2_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

GPT_MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --attention-softmax-in-fp32 
"
# 因为 tensor-model-parallel-size 1 pipeline-model-parallel-size 1 而且 GPUS_PER_NODE=4 
# 评估出来的data parallel size 是 4
# global-batch-size必须是 micro-batch-size * data parallel size 的整数倍

#  --micro-batch-size 4 \ 
#  --global-batch-size 32 \ 
#  在core_r0.8.0 中这两个参数是 micro-batch-size 1 和 global-batch-size 1536 导致训练的step走的很慢
#  需要研究下这两个参数的含义
#  
#  --rampup-batch-size 16 16 5859375 
#  --adam-beta1 0.9 
#  --adam-beta2 0.95 
#  这三个参数我给删除了，需要研究一下这几个改动的参数对整个训练的影响

TRAINING_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 3000 \
    --weight-decay 0.1 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --fp16 \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 
"
# 不进行模型并行
# 这里使用的是 1dp 1pp 4gpus 

#  这两个参数在训练的时候感觉速度也不快，需要研究下
MODEL_PARALLEL_ARGS="
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 
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
    ${GPT_MODEL_ARGS} \
    ${TRAINING_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    ${DATA_ARGS} \
    ${EVAL_AND_LOGGING_ARGS}
