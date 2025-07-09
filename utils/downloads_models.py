"""
下载预训练模型脚本
"""
import os
from ultralytics import YOLO


def download_yolov10_models():
    """下载YOLOv10预训练模型"""
    models = [
        'yolov10n.pt',  # Nano
        'yolov10s.pt',  # Small
        'yolov10m.pt',  # Medium
        'yolov10b.pt',  # Balanced
        'yolov10l.pt',  # Large
        'yolov10x.pt',  # Extra Large
    ]

    os.makedirs('weights', exist_ok=True)

    for model_name in models:
        print(f"下载 {model_name}...")
        try:
            # 创建YOLO对象会自动下载模型
            model = YOLO(model_name)
            print(f"✓ {model_name} 下载完成")
        except Exception as e:
            print(f"✗ {model_name} 下载失败: {e}")

    print("所有模型下载完成!")


if __name__ == '__main__':
    download_yolov10_models()