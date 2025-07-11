import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

def xywh_to_yolo(bbox, img_width, img_height):
    """
    将XYWH格式转换为YOLO格式

    Args:
        bbox: [x, y, w, h] 像素坐标
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        [x_center, y_center, width, height] 归一化坐标
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]


def xyxy_to_yolo(bbox, img_width, img_height):
    """
    将XYXY格式转换为YOLO格式

    Args:
        bbox: [x1, y1, x2, y2] 像素坐标
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        [x_center, y_center, width, height] 归一化坐标
    """
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]


def yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """
    将YOLO格式转换为XYXY格式

    Args:
        yolo_bbox: [x_center, y_center, width, height] 归一化坐标
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        [x1, y1, x2, y2] 像素坐标
    """
    x_center, y_center, width, height = yolo_bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]


def create_yolo_annotation(image_path, bboxes, class_ids, output_path):
    """
    创建YOLO格式的标注文件

    Args:
        image_path: 图像路径
        bboxes: 边框列表 [[x1,y1,x2,y2], ...]
        class_ids: 类别ID列表 [0, 1, 0, ...]
        output_path: 输出标注文件路径
    """
    # 读取图像获取尺寸
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # 转换并保存标注
    with open(output_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_ids):
            yolo_bbox = xyxy_to_yolo(bbox, img_width, img_height)
            line = f"{class_id} {' '.join(map(str, yolo_bbox))}\n"
            f.write(line)

    print(f"标注文件已保存: {output_path}")


def coco_to_yolo(coco_json_path, output_dir, images_dir):
    """将COCO格式转换为YOLO格式"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_ids = list(categories.keys())
    category_ids.sort()

    # 创建图像映射
    images = {img['id']: img for img in coco_data['images']}

    # 处理标注
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # 转换并保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_id, anns in annotations_by_image.items():
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']

        # 创建YOLO标注文件
        yolo_annotations = []
        for ann in anns:
            category_id = ann['category_id']
            class_id = category_ids.index(category_id)

            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # 转换为YOLO格式
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # 保存标注文件
        label_file = output_path / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    # 创建类别映射文件
    with open(output_path / 'classes.txt', 'w') as f:
        for cat_id in category_ids:
            f.write(f"{categories[cat_id]}\n")

    print(f"转换完成! 输出目录: {output_dir}")
    print(f"类别数量: {len(categories)}")


def voc_xml_to_yolo(xml_dir, output_dir, images_dir):
    """将VOC XML格式转换为YOLO格式"""
    xml_files = list(Path(xml_dir).glob('*.xml'))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_classes = set()

    # 第一遍：收集所有类别
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            all_classes.add(class_name)

    # 创建类别到ID的映射
    class_to_id = {cls: idx for idx, cls in enumerate(sorted(all_classes))}

    # 第二遍：转换标注
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_annotations = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_to_id[class_name]

            # 获取边框坐标
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 转换为YOLO格式
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # 保存标注文件
        label_file = output_path / f"{xml_file.stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    # 保存类别映射
    with open(output_path / 'classes.txt', 'w') as f:
        for class_name in sorted(all_classes):
            f.write(f"{class_name}\n")

    print(f"转换完成! 输出目录: {output_dir}")
    print(f"类别数量: {len(all_classes)}")
    print(f"类别: {sorted(all_classes)}")


if __name__ == "__main__":
    # COCO转YOLO示例
    # coco_to_yolo("annotations/instances_train2017.json", "labels/train", "images/train")

    # VOC转YOLO示例
    voc_xml_to_yolo(r"F:\Dataset\Detection\SSDD\Annotations", r"F:\Dataset\Detection\SSDD\labels_yolo", "JPEGImages")

    # image_path = "data/train/images/example.jpg"
    # bboxes = [[100, 50, 300, 200], [400, 150, 600, 350]]  # XYXY格式
    # class_ids = [0, 1]  # 对应的类别ID
    # output_path = "data/train/labels/example.txt"
    #
    # create_yolo_annotation(image_path, bboxes, class_ids, output_path)