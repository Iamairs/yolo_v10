"""
数据处理工具
"""
import os
import shutil
import random
from pathlib import Path
import yaml


def create_dataset_yaml(train_path, val_path, test_path, class_names, save_path):
    """创建数据集配置文件"""
    dataset_config = {
        'path': str(Path(save_path).parent),  # 数据集根目录
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),  # 类别数量
        'names': class_names  # 类别名称
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    print(f"数据集配置文件已保存: {save_path}")
    return dataset_config


def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """分割数据集"""
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # 创建输出目录
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    image_files = [f.stem for f in image_files]  # 获取文件名（不含扩展名）

    # 随机打乱
    random.shuffle(image_files)

    # 计算分割点
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 分割
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # 复制文件
    for split_name, file_list in splits.items():
        for filename in file_list:
            # 复制图像
            for ext in ['.jpg', '.png', '.jpeg']:
                src_img = Path(image_dir) / f"{filename}{ext}"
                if src_img.exists():
                    dst_img = output_path / split_name / 'images' / f"{filename}{ext}"
                    shutil.copy2(src_img, dst_img)
                    break

            # 复制标签
            src_label = Path(label_dir) / f"{filename}.txt"
            if src_label.exists():
                dst_label = output_path / split_name / 'labels' / f"{filename}.txt"
                shutil.copy2(src_label, dst_label)

    print(f"数据集分割完成:")
    print(f"训练集: {len(splits['train'])} 张")
    print(f"验证集: {len(splits['val'])} 张")
    print(f"测试集: {len(splits['test'])} 张")


def check_dataset(data_yaml_path):
    """检查数据集完整性"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_config['path'])

    for split in ['train', 'val', 'test']:
        if split in data_config:
            split_path = base_path / data_config[split]
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'

            if not images_path.exists():
                print(f"警告: {images_path} 不存在")
                continue

            if not labels_path.exists():
                print(f"警告: {labels_path} 不存在")
                continue

            # 统计文件数量
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            label_files = list(labels_path.glob('*.txt'))

            print(f"{split}: {len(image_files)} 图像, {len(label_files)} 标签")

            # 检查是否有孤立文件
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}

            orphan_images = image_stems - label_stems
            orphan_labels = label_stems - image_stems

            if orphan_images:
                print(f"  警告: {len(orphan_images)} 个图像没有对应标签")
            if orphan_labels:
                print(f"  警告: {len(orphan_labels)} 个标签没有对应图像")