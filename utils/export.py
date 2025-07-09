"""
YOLOv10模型导出脚本
"""
import argparse
from ultralytics import YOLO
import os


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv10 Model Export')
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--format', type=str, default='torchscript',
                        choices=['torchscript', 'onnx', 'tflite'],
                        help='导出格式')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--half', action='store_true',
                        help='使用FP16半精度')
    parser.add_argument('--dynamic', action='store_true',
                        help='动态轴')
    parser.add_argument('--simplify', action='store_true',
                        help='简化ONNX模型')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset版本')
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查模型文件
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    # 加载模型
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")

    # 导出参数
    export_args = {
        'format': args.format,
        'imgsz': args.imgsz,
        'half': args.half,
        'dynamic': args.dynamic,
    }

    # 针对ONNX的特殊参数
    if args.format == 'onnx':
        export_args.update({
            'simplify': args.simplify,
            'opset': args.opset,
        })

    print(f"开始导出为 {args.format} 格式...")
    print(f"导出参数: {export_args}")

    # 导出模型
    exported_model = model.export(**export_args)

    print(f"导出完成: {exported_model}")


if __name__ == '__main__':
    main()