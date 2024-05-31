# DATA620004：神经网络和深度学习作业1

复旦大学 神经网络和深度学习  HW1

任务：使用仅 numpy 构建的三层神经网络，在 fashion-mnist 数据集上实现图像分类。

## 文件结构
- `Homework_main.py`：包括数据载入、模型训练、参数权重存储以及测试的完整流程。
- `Homework_search.py`：进行参数网格搜索，评估不同参数组合的模型训练效果。
- `weights_visual.py`：训练权重参数的可视化。

## 快速开始

### 安装依赖

安装必需的 Python 库：

```bash
pip install numpy matplotlib tqdm
```

### 数据准备

确保将数据集存放在以下目录中：

```
/fashion-mnist/
```

### 训练模型

1. 打开 `Homework_main.py`，根据需求调整网络设置和 `train_my_network()` 函数的参数。
2. 执行以下命令以启动训练：

```bash
python Homework_main.py
```

训练完成后，模型的损失图像和验证集准确率图像将被保存到指定目录。

## 测试模型

在模型参数保存之后，系统会自动执行测试并返回测试集上的准确率结果。您也可以通过加载保存的参数并调用测试函数进行单独测试。

### 权重可视化

运行 `weights_visual.py` 以读取并可视化训练过的权重参数：

```bash
python weights_visual.py
```
