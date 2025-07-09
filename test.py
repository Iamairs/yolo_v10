import argparse
from ultralytics import YOLO
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv10 Testing')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                        help='数据集配置文件')
    parser.add_argument('--split', type=str, default='test',
                        help='测试数据集分割 (test/val)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--batch', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存结果为txt格式')
    parser.add_argument('--save-conf', action='store_true',
                        help='保存置信度')
    parser.add_argument('--project', type=str, default='runs/test',
                        help='保存路径')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    # 创建保存目录
    os.makedirs(args.project, exist_ok=True)

    # 加载模型
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")

    # 测试参数
    test_args = {
        'data': args.data,
        'split': args.split,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'conf': args.conf,
        'iou': args.iou,
        'device': args.device,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'project': args.project,
        'name': args.name,
        'verbose': True,
        'plots': True,
    }

    print("开始测试...")
    print(f"测试参数: {test_args}")

    # 运行测试
    results = model.val(**test_args)

    # 打印结果
    print("\n=== 测试结果 ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

    print(f"\n结果保存在: {results.save_dir}")


if __name__ == '__main__':
    main()