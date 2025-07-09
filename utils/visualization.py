"""
可视化工具
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def load_class_names(data_yaml_path):
    """从数据集配置文件加载类别名称"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    return data_config['names']


def draw_predictions(image, results, class_names=None, conf_threshold=0.25):
    """在图像上绘制预测结果"""
    img = image.copy()

    if len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                # 获取边框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                if conf >= conf_threshold:
                    # 绘制边框
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # 绘制标签
                    if class_names:
                        label = f"{class_names[cls]}: {conf:.2f}"
                    else:
                        label = f"Class {cls}: {conf:.2f}"

                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10),
                                  (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(img, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img


def visualize_dataset_samples(data_yaml_path, split='train', num_samples=6):
    """可视化数据集样本"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_config['path'])
    split_path = base_path / data_config[split]
    images_path = split_path / 'images'
    labels_path = split_path / 'labels'
    class_names = data_config['names']

    # 获取图像文件
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    image_files = image_files[:num_samples]

    # 创建子图
    rows = (num_samples + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, img_path in enumerate(image_files):
        row = i // 3
        col = i % 3

        # 读取图像
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 读取标签
        label_path = labels_path / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 绘制边框
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # 转换为像素坐标
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    # 绘制边框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # 添加类别标签
                    if cls_id < len(class_names):
                        label = class_names[cls_id]
                        cv2.putText(img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示图像
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{img_path.name}")
        axes[row, col].axis('off')

    # 隐藏多余的子图
    for i in range(num_samples, rows * 3):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_curves(results_dir):
    """绘制训练曲线"""
    results_file = Path(results_dir) / 'results.csv'

    if not results_file.exists():
        print(f"结果文件不存在: {results_file}")
        return

    # 读取结果
    import pandas as pd
    df = pd.read_csv(results_file)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # mAP曲线
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    axes[1, 0].set_title('mAP')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Precision & Recall
    axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()