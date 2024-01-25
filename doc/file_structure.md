# pytorch-segmentation

## 文件结构：
```

├─config --- 模型训练参数配置
├─dataloaders
│  └─datasets --- 自定义数据集
├─doc
├─model
│  ├─transunet
│  └─unet
└─utils
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取DRIVE数据集(视网膜血管分割)
  ├── train.py: 以单GPU为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── compute_mean_std.py: 统计数据集各通道的均值和标准差
```