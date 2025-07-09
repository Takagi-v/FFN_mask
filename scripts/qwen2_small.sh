#!/bin/bash

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# --- 参数设置 (基于 qwen2_xnli.sh，为本地测试调整) ---
LM_MODEL_PATH="/public/share/model/Qwen2.5-3B-Instruct" # 改为绝对路径
TRAIN_DATA_PATH="/public/home/sjtu_intern/users/yilu.cao/dataset/XNLI-15way/xnli.15way.orig.tsv" # 修改为绝对路径
DEV_DATA_PATH="/public/home/sjtu_intern/users/yilu.cao/dataset/XNLI-15way/xnli.15way.orig.tsv" 
OUTPUT_DIR="./output/qwen2-3b/xnli_local_test"         # 使用新的测试输出目录
NUM_GPUS=1

cd ../
# --- 环境设置结束 ---

# --- 训练命令 (修改了以减少显存使用) ---
WANDB_MODE=offline python train_mask.py \
    --lm_model_path $LM_MODEL_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --dev_data_path $DEV_DATA_PATH \
    --model_type qwen2 \
    --attn_implementation "eager" \
    --seed 42 \
    --dataset_use_cache false \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --max_steps 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 2 \
    --save_only_model \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --logging_dir ./logs \
    --logging_steps 5 \
    --learning_rate 1e-2 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-4}' \
    --tf32 false \
    --ddp_find_unused_parameters false \
    --weight_decay 0 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 1 \
    --use_gumbel \
    --gumbel_hard \
    --tau_temp_begin 4 \
    --tau_decay_steps 0.6 \
    --tau_temp_end 0.05 \
    --norm_lambda 0 \
    --norm_power 1 \
    --report_to none \
    --fp16 \

echo "Local test script finished."