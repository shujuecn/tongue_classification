#!/bin/bash

# 模型列表
models=("resnet18" "resnet50" "vgg16" "vgg19" "alexnet")


# 循环遍历模型和批次大小
for model in "${models[@]}"
do

    python train.py --model "$model" --batch_size 64 --num_epochs 70 --learning_rate 0.0001 --random_seed 31
    echo "-----------------------------"
    echo ""

done

python send_email.py

# while :; do /usr/sbin/shutdown; sleep 3s; done
