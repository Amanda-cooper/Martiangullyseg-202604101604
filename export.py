from unet import Unet  # 导入第二个文件中的Unet工具类

# 配置参数 (与训练时一致)
config = {
    "model_path": 'D:/unet-pytorch-main/logs/UNet.pth',
    "num_classes": 2,
    "backbone": "vgg",
    "input_shape": [640, 640],
    "cuda": True
}

# 初始化UNet工具
unet_tool = Unet(**config)

# 导出ONNX模型
output_path = '/logs/unet_best.onnx'
unet_tool.convert_to_onnx(
    simplify=True,
    model_path=output_path
)

print(f"ONNX模型已成功导出到: {output_path}")