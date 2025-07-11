import os
import yaml
from ultralytics import YOLO
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv10 Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='训练配置文件路径')
    parser.add_argument('--dataset_name', type=str, default='HRSID',
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='configs/dataset',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='yolov10s.pt',
                        help='预训练模型路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=800,
                        help='图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0, 1, 2, ... 或 cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='项目保存路径')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练路径')
    return parser.parse_args()


def load_config(config_path):
    """加载训练配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()

    recording_dir = Path(args.project)
    if not recording_dir.exists():
        print(f"创建目录: {recording_dir}")
        recording_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(f'weights/weight_{args.dataset_name}')
    if not weights_dir.exists():
        print(f"创建目录: {weights_dir}")
        weights_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    config = load_config(args.config)

    # 初始化模型
    if args.resume:
        model = YOLO(args.resume)
        print(f"从 {args.resume} 恢复训练")
    else:
        model = YOLO(args.model)
        print(f"使用预训练模型: {args.model}")

    # 训练参数
    train_args = {
        'data': f"{args.data_dir}/{args.dataset_name}.yaml",  # 数据集配置文件路径
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'val': True,
        'plots': True,
        'verbose': True,
    }

    # 合并配置文件中的参数
    train_args.update(config)

    print("开始训练...")
    print(f"训练参数: {train_args}")

    # 开始训练
    results = model.train(**train_args)

    # 保存最终模型
    model.save(f'{weights_dir}/yolov10_final.pt')
    print("训练完成!")
    print(f"最佳模型保存在: {recording_dir}/weights/best.pt")


if __name__ == '__main__':
    main()