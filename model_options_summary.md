# 模型选项总结

## 可用的模型类型

在`train.py`中，你可以通过设置`model_type`参数来选择不同的模型：

### 1. UNet_origin

- **描述**: 原版 UNet（经典编码器-解码器结构）
- **特点**:
  - 使用自定义卷积层作为编码器
  - 不依赖预训练骨干网络
  - 支持可选的注意力机制
  - 从零开始训练
- **参数**:
  ```python
  model_type = "UNet_origin"
  n_channels = 3              # 输入通道数
  bilinear = False           # 双线性插值上采样
  c_attention = False        # 通道注意力
  s_attention = False        # 空间注意力
  ```

### 2. UNet_origin_Attention1

- **描述**: 原版 UNet + 连接处注意力机制
- **特点**:
  - 在跳跃连接处添加注意力机制
  - 使用 CFBlock 注意力模块
  - 从零开始训练
- **参数**:
  ```python
  model_type = "UNet_origin_Attention1"
  n_channels = 3              # 输入通道数
  bilinear = False           # 双线性插值上采样
  attention = True           # 启用注意力机制
  ```

### 3. UNet_rigin_Attention2

- **描述**: 原版 UNet + 卷积层注意力机制
- **特点**:
  - 在卷积层中添加注意力机制
  - 使用 DoubleConvAttention 模块
  - 从零开始训练
- **参数**:
  ```python
  model_type = "UNet_rigin_Attention2"
  n_channels = 3              # 输入通道数
  bilinear = False           # 双线性插值上采样
  ```

### 4. Unet

- **描述**: 改进版 UNet（使用预训练骨干网络）
- **特点**:
  - 使用预训练骨干网络作为编码器
  - 支持多种骨干网络（VGG、ResNet50、ConvNeXt 等）
  - 更好的特征表示能力
  - 需要预训练权重
- **参数**:
  ```python
  model_type = "Unet"
  backbone = "vgg"           # 骨干网络选择
  pretrained = True          # 使用预训练权重
  ```

### 5. UnetAttention1

- **描述**: 带注意力的 UNet（集成 CBAM 注意力机制）
- **特点**:
  - 在解码器中集成 CBAM 注意力机制
  - 使用预训练骨干网络
  - 结合了改进版 UNet 和注意力机制的优势
- **参数**:
  ```python
  model_type = "UnetAttention1"
  backbone = "vgg"           # 骨干网络选择
  pretrained = True          # 使用预训练权重
  ```

## 模型对比

| 特性       | UNet_origin  | UNet_origin_Attention1 | UNet_rigin_Attention2  | Unet           | UnetAttention1   |
| ---------- | ------------ | ---------------------- | ---------------------- | -------------- | ---------------- |
| 架构类型   | 经典 UNet    | 经典 UNet+连接处注意力 | 经典 UNet+卷积层注意力 | 改进 UNet      | 改进 UNet+注意力 |
| 编码器     | 自定义卷积层 | 自定义卷积层           | 自定义卷积层           | 预训练骨干网络 | 预训练骨干网络   |
| 注意力机制 | 可选         | CFBlock                | DoubleConvAttention    | 无             | 集成 CBAM        |
| 预训练权重 | 不需要       | 不需要                 | 不需要                 | 需要           | 需要             |
| 参数量     | 较少         | 中等                   | 中等                   | 较多           | 最多             |
| 训练难度   | 较难         | 较难                   | 较难                   | 较易           | 较易             |
| 性能       | 基础         | 较好                   | 较好                   | 较好           | 最好             |

## 使用建议

### 选择 UNet_origin 的情况：

- 需要经典的 UNet 实现
- 显存有限
- 想要从零开始训练
- 研究用途

### 选择 UNet_origin_Attention1 的情况：

- 需要经典 UNet + 连接处注意力
- 显存有限但想要注意力机制
- 从零开始训练
- 对跳跃连接注意力感兴趣

### 选择 UNet_rigin_Attention2 的情况：

- 需要经典 UNet + 卷积层注意力
- 显存有限但想要注意力机制
- 从零开始训练
- 对卷积层注意力感兴趣

### 选择 Unet 的情况：

- 需要更好的性能
- 有足够的显存
- 可以使用预训练权重
- 生产环境

### 选择 UnetAttention1 的情况：

- 需要最佳性能
- 有足够的计算资源
- 对注意力机制感兴趣
- 复杂的分割任务

## 配置示例

### 训练 UNet_origin：

```python
model_type = "UNet_origin"
n_channels = 3
bilinear = False
c_attention = False
s_attention = False
```

### 训练 UNet_origin_Attention1：

```python
model_type = "UNet_origin_Attention1"
n_channels = 3
bilinear = False
attention = True
```

### 训练 UNet_rigin_Attention2：

```python
model_type = "UNet_rigin_Attention2"
n_channels = 3
bilinear = False
```

### 训练 Unet：

```python
model_type = "Unet"
backbone = "vgg"
pretrained = True
```

### 训练 UnetAttention1：

```python
model_type = "UnetAttention1"
backbone = "vgg"
pretrained = True
```
