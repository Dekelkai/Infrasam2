# 数据集和模型权重

验证数据，下载数据集解压到Infrasam2\notebooks\tzb_data\val 中： https://pan.baidu.com/s/1rNvtDVKeSClj8Y_cZm-Sgg?pwd=x275 提取码: x275 

微调数据（1）：finetuning-data_val1_train.zip 链接: https://pan.baidu.com/s/1Ij8DIfGBsWNr7q2gSPWTvA?pwd=grx7 提取码: grx7 

微调数据（2）：AntiUAV-train-finetune.zip 链接: https://pan.baidu.com/s/1XBqfnMS0ro2R27LBl4imuA?pwd=u5dp 提取码: u5dp 



模型权重放在sam2_logs/configs/*/checkpoints/中

AntiAUVdata_checkpoint.py：链接: https://pan.baidu.com/s/1mR7KEC1HxKfBny81RMP-sQ?pwd=wgtt 提取码: wgtt 

tzb_data_checkpoint.pt：链接: https://pan.baidu.com/s/1iQTK6MO3WYmWTrI2SUtoJQ?pwd=5jaa 提取码: 5jaa 

# 环境安装

pip install -e ".[notebooks]"

# 一阶段微调
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_tzb_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8

# 二阶段微调
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_with_finetune_AntiUAV.yaml.yaml \
    --use-cluster 0 \
    --num-gpus 8

# 模型预测
在notebook/run.ipynb中进行配置和测试
