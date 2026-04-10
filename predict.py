#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os

import cv2
import numpy as np
from PIL import Image
import torch  # 新增：导入torch

# 假设你的UNet_origin定义在unet.py中，需要确保能正确导入
from unet import Unet_ONNX, Unet, UNet_origin  # 新增：导入UNet_origin

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测
    #   'video'             表示视频检测
    #   'fps'               表示测试fps
    #   'dir_predict'       表示遍历文件夹进行检测并保存
    #   'export_onnx'       表示将模型导出为onnx
    #   'predict_onnx'      表示利用导出的onnx模型进行预测
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类
    #-------------------------------------------------------------------------#
    count           = True
    name_classes    = ["background","Gully"]
    #----------------------------------------------------------------------------------------------------------#
    #   video相关参数（仅mode='video'有效）
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   fps测试相关参数（仅mode='fps'有效）
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/CESP0116721395A0104.jpg"
    #-------------------------------------------------------------------------#
    #   文件夹预测相关参数（仅mode='dir_predict'有效）
    #-------------------------------------------------------------------------#
    model_type  = "UNet_origin"  # 明确使用标准UNet
    model_path = "/root/autodl-tmp/UNET/logs/UNet.pth"
    dir_origin_path = "/root/autodl-tmp/UNET/img/"
    dir_save_path   = "/root/autodl-tmp/UNET/img_out/"
    num_classes = len(name_classes)  # 新增：根据类别数确定输出通道数
    #-------------------------------------------------------------------------#
    #   onnx导出相关参数
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    # ===================== 关键修改1：初始化正确的模型结构 =====================
    if mode != "predict_onnx":
        # 初始化标准UNet_origin模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet = UNet_origin(n_channels=3, n_classes=num_classes, bilinear=False)
        # 加载权重（添加weights_only=True解决警告，同时适配新torch版本）
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        unet.load_state_dict(state_dict)
        unet = unet.eval()  # 设置为评估模式
        unet = unet.to(device)
    else:
        yolo = Unet_ONNX()

    # ===================== 关键修改2：适配UNet_origin的detect_image方法 =====================
    # 新增：定义适配UNet_origin的检测函数（如果你的Unet类有detect_image，这里需要对齐逻辑）
    def detect_image_unet_origin(model, image, count=False, name_classes=None):
        device = next(model.parameters()).device
        # 图像预处理
        img = np.array(image)
        img = cv2.resize(img, (1024, 1024))  # 根据你的训练输入尺寸调整
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # 绘制分割结果（和原detect_image逻辑对齐）
        seg_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        # 定义颜色：background为黑色，Gully为红色
        colors = [(0, 0, 0), (255, 0, 0)]
        for c in range(num_classes):
            seg_img[pred == c] = colors[c]
        
        # 混合原图和分割图
        image = np.array(image)
        image = cv2.resize(image, (pred.shape[1], pred.shape[0]))
        blend_img = cv2.addWeighted(image, 0.7, seg_img, 0.3, 0)
        
        # 计数逻辑（如果需要）
        if count and name_classes is not None:
            for c in range(1, num_classes):  # 跳过background
                area = np.sum(pred == c)
                ratio = area / (pred.shape[0] * pred.shape[1])
                print(f"{name_classes[c]}: 像素数={area}, 占比={ratio:.4f}")
        
        return Image.fromarray(blend_img)

    # 替换原unet.detect_image为新的适配函数
    unet.detect_image = lambda img, count=False, name_classes=None: detect_image_unet_origin(unet, img, count, name_classes)

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(unet.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)  # 修复：使用指定的fps测试图片路径
        tact_time = 0.0
        for _ in range(test_interval):
            t1 = time.time()
            unet.detect_image(img)
            tact_time += time.time() - t1
        tact_time /= test_interval
        print(f"{tact_time:.4f} seconds, {1/tact_time:.2f} FPS, @batch_size 1")
        
    elif mode == "dir_predict":
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        # 如果需要导出onnx，需要额外处理
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        torch.onnx.export(
            unet,
            dummy_input,
            onnx_save_path,
            opset_version=12,
            simplify=simplify,
            input_names=["input"],
            output_names=["output"]
        )
        print(f"ONNX模型已保存到: {onnx_save_path}")
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")