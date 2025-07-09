"""
YOLOv10推理脚本
"""
import argparse
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv10 Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源 (图像/视频/文件夹路径)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--save', action='store_true',
                        help='保存结果')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存预测结果为txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='保存置信度')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='保存路径')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--line-thickness', type=int, default=3,
                        help='边框线条粗细')
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查模型和输入源
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"输入源不存在: {args.source}")

    # 创建保存目录
    if args.save:
        os.makedirs(args.project, exist_ok=True)

    # 加载模型
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")

    # 推理参数
    predict_args = {
        'source': args.source,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'device': args.device,
        'save': args.save,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'show': args.show,
        'project': args.project,
        'name': args.name,
        'line_width': args.line_thickness,
        'verbose': True,
    }

    print("开始推理...")
    print(f"推理参数: {predict_args}")

    # 运行推理
    results = model.predict(**predict_args)

    print(f"\n推理完成!")
    if args.save:
        print(f"结果保存在: {Path(args.project) / args.name}")


if __name__ == '__main__':
    main()