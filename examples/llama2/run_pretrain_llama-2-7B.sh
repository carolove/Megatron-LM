TP=1
PP=1
 
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
NNODES=${WORLD_SIZE}
WORK_PATH=/workspace 
#export NCCL_SOCKET_IFNAME=eth,en,em,bond

export NCCL_SOCKET_IFNAME=bond1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO
export NCCL_DEBUG=
export CUDA_LAUNCH_BLOCKING=1
#export LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=$WORK_PATH/cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export PYTHONPATH=${TOOL_PATH}:${PYTHONPATH}
export OMP_NUM_THREADS=20
export NCCL_IB_GID_INDEX=3
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 7 B
HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
NUM_LAYERS=32 # e.g. llama-13b: 40
NUM_HEADS=32 # e.g. llama-13b: 40
SEQ_LENGTH=4096
NUM_KV_HEADS=32 # llama2 70B uses GQA

CODE_PATH=/workspace/Megatron-LM
PREMODEL=/workspace/models/Llama-2-7b-hf-to-megatron-tp1-pp1
CHECKPOINT_PATH=/workspace/models/Llama-2-7b-hf-to-megatron-tp1-pp1-out
TOKENIZER_MODEL=/workspace/models/llama-2-7b/tokenizer.model
TENSORBOARD_LOGS_PATH=/workspace/tensorboard_logs
DATA_PATH=/workspace/data/my-gpt2_text_document


MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8 
TRAIN_STEPS=2 
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

torchrun $DISTRIBUTED_ARGS \
       ${CODE_PATH}/pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $PREMODEL \
       --train-data-path $DATA_PATH \
       --valid-data-path $DATA_PATH \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 1 \
       --eval-interval 10000000 \
       --eval-iters 1000 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --exit-on-missing-checkpoint \
       --use-checkpoint-args \
       --no-load-optim \
       --no-load-rng \
       --fp16 \
       --untie-embeddings-and-output-weights \
       --use-rotary-position-embeddings \
       --normalization RMSNorm \
       --no-position-embedding \
       --no-masked-softmax-fusion \
       --use-flash-attn \
       --make-vocab-size-divisible-by 512 \
       --ckpt-format torch \
       --use-legacy-models \
       --attention-softmax-in-fp32