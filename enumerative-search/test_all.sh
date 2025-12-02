#!/bin/bash

# 比如我们测试前 10 道题 (0 到 9)
for i in {0..400}
do
    echo "================ Testing Problem $i ================"
    
    # 1. 先生成这道题的预测文件 (需要调用你的 predict.py)
    # 注意：确保 predict.py 的路径是对的
    uv run ../transformer/predict.py \
        --model ../transformer/models/deepcoder_transformer/checkpoint-129398 \
        --dataset eval_set.pickle \
        --problem-idx $i \
        --output-dir data/example/predictions > /dev/null 2>&1

    # 2. 运行 Baseline (不使用神经网络)
    echo "--- Baseline ---"
    uv run ./search example 3 3 $i 0 -1 | grep "Nodes explored"

    # 3. 运行 Neural Network (使用神经网络)
    echo "--- DeepCoder (NN) ---"
    uv run ./search example 3 3 $i 1 -1 | grep "Nodes explored"
    
    echo ""
done