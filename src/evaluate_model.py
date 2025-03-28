import os
import json
import xml.etree.ElementTree as ET
import time

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
import numpy as np
from collections import defaultdict
import tensorflow as tf
import matplotlib.pyplot as plt


# List to hold the prediction results in COCO format
coco_predictions = {
    "images": [],
    "annotations": [],
    "categories": [],
}


### Create Ground Truth #####
def read_classes_from_file(class_file):
    """
    Read the classes from the text file and return a list of class names.
    The class IDs will be assigned based on the order of classes in this list.
    """
    with open(class_file, 'r') as f:
        classes = f.readlines()
    # Remove any extra spaces or newline characters
    return [cls.strip() for cls in classes]


def convert_voc_to_coco(voc_annotations_path, image_folder_path, class_file, output_folder):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Read classes from the class file (order matters)
    class_names = read_classes_from_file(class_file)
    category_map = {class_name: idx + 1 for idx, class_name in enumerate(class_names)}  # Start class IDs from 1

    annotation_id = 1
    image_id = 1

    # Iterate over XML files in the VOC annotations folder
    for xml_file in os.listdir(voc_annotations_path):
        if not xml_file.endswith('.xml'):
            continue
        
        # Parse XML file
        tree = ET.parse(os.path.join(voc_annotations_path, xml_file))
        root = tree.getroot()
        
        # Get image information
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        # Add image information to COCO
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_format["images"].append(image_info)

        # Process annotations
        for obj in root.findall('object'):
            category_name = obj.find('name').text
            if category_name in category_map:
                category_id = category_map[category_name]
            else:
                continue  # Skip objects whose category isn't in the class file

            # Get bounding box and object details
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Add annotation info to COCO format
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            }
            coco_format["annotations"].append(annotation_info)
            annotation_id += 1

        image_id += 1

    # Add category information to COCO format
    for class_name, class_id in category_map.items():
        coco_format["categories"].append({
            "id": class_id,
            "name": class_name
        })

    # Save COCO annotations to a JSON file
    with open(os.path.join(output_folder, 'annotations_coco.json'), 'w') as json_file:
        json.dump(coco_format, json_file)


### Create Predictions###

# Helper functions for drawing and saving images (unchanged)
def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


# Prepare for generating the predictions JSON
def process_images(model_filename, labels_filename, images_path, threshold=0.4, count=5):
    labels = read_label_file(labels_filename) if labels_filename else {}
    
    # Use TensorFlow to load the TFLite model (no EdgeTPU)
    interpreter = tf.lite.Interpreter(model_path=model_filename)
    interpreter.allocate_tensors()

    category_map = {}  # to map label ID to category ID
    annotation_id = 1
    image_id = 1

    # Populate categories list based on the labels file
    for label in labels.values():
        category_map[label] = len(category_map) + 1
        coco_predictions['categories'].append({
            "id": category_map[label],
            "name": label,
        })

    # Loop over all images in the test folder
    for filename in os.listdir(images_path):
        image_path = os.path.join(images_path, filename)
        image = Image.open(image_path)
        _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.Resampling.LANCZOS))

        print(f'Processing image: {filename}')

        for _ in range(count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, threshold, scale)
            print('%.2f ms' % (inference_time * 1000))

        # If no objects detected, skip to the next image
        if not objs:
            continue

        # Add image metadata to the 'images' list in COCO format
        image_info = {
            "id": image_id,
            "file_name": filename,
            "width": image.width,
            "height": image.height,
        }
        coco_predictions["images"].append(image_info)

        # Process detections and add annotations to the 'annotations' list in COCO format
        for obj in objs:
            category_name = labels.get(obj.id, str(obj.id))
            category_id = category_map[category_name]

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(obj.bbox.xmin), float(obj.bbox.ymin), float(obj.bbox.xmax - obj.bbox.xmin), float(obj.bbox.ymax - obj.bbox.ymin)],
                "score": obj.score,
                "iscrowd": 0,  # No crowd information in this case, so it's set to 0
                "area": (obj.bbox.xmax - obj.bbox.xmin) * (obj.bbox.ymax - obj.bbox.ymin),
            }

            coco_predictions["annotations"].append(annotation_info)
            annotation_id += 1

        image_id += 1
        
    # Save the COCO predictions to a JSON file
    # Ensure annotations is a list of objects
    prediction_json = []
    for annotation in coco_predictions["annotations"]:
        prediction_json.append({
            "image_id": annotation["image_id"],
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "score": annotation["score"],
            "iscrowd": annotation["iscrowd"],
            "area": annotation["area"],
            "id": annotation["id"]
        })

    # Write the predictions JSON to file
    with open('predictions_coco.json', 'w') as f:
        json.dump(prediction_json, f)


def process_image(interpreter, image_path, labels, category_map, threshold=0.4, count=5):
    filename = os.path.basename(image_path)
    image = Image.open(image_path)
    _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.Resampling.LANCZOS))
    
    for _ in range(count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, threshold, scale)

    if not objs:
        return None  # Skip images without detections

    image_info = {
        "file_name": filename,
        "width": image.width,
        "height": image.height,
    }
    annotations = []
    
    for obj in objs:
        category_name = labels.get(obj.id, str(obj.id))
        category_id = category_map[category_name]

        annotation_info = {
            "category_id": category_id,
            "bbox": [float(obj.bbox.xmin), float(obj.bbox.ymin), float(obj.bbox.xmax - obj.bbox.xmin), float(obj.bbox.ymax - obj.bbox.ymin)],
            "score": obj.score,
            "iscrowd": 0,
            "area": (obj.bbox.xmax - obj.bbox.xmin) * (obj.bbox.ymax - obj.bbox.ymin),
        }
        annotations.append(annotation_info)

    return image_info, annotations



def process_images_parallel(model_filename, labels_filename, images_path, output_folder, threshold=0.4, count=5, num_workers=4):
    print('started evaluation of Test Data with the .tflite model')
    model_filename = os.path.join(output_folder, model_filename)
    labels_filename = os.path.join(output_folder, labels_filename)
    labels = read_label_file(labels_filename) if labels_filename else {}

    strategy = tf.distribute.MirroredStrategy()  # Use multiple GPUs
    category_map = {}
    coco_predictions = {"images": [], "annotations": [], "categories": []}
    
    annotation_id = 1

    # Map label names to category IDs
    for label in labels.values():
        category_map[label] = len(category_map) + 1
        coco_predictions['categories'].append({"id": category_map[label], "name": label})

    image_paths = [os.path.join(images_path, f) for f in os.listdir(images_path)]

    def worker(image_path):
        with strategy.scope():
            interpreter = tf.lite.Interpreter(model_path=model_filename)
            interpreter.allocate_tensors()
            return process_image(interpreter, image_path, labels, category_map, threshold, count)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker, image_paths))

    image_id = 1
    for result in results:
        if result:
            image_info, annotations = result
            image_info["id"] = image_id
            coco_predictions["images"].append(image_info)

            for annotation in annotations:
                annotation["image_id"] = image_id
                annotation["id"] = annotation_id
                coco_predictions["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    with open(os.path.join(output_folder, 'predictions_coco.json'), 'w') as f:
        json.dump(coco_predictions["annotations"], f)
    
    print('finished .tflite model evaluation')


### Evaluation###
def load_class_names(class_names_file):
    """
    Load class names from a text file. The file should have one class name per line.
    The line number corresponds to the class ID, starting from 1.
    """
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def calculate_precision_recall_for_class(ground_truth, predictions, iou_threshold=0.5):
    """
    Calculate precision and recall for a class based on sorted predictions.
    This will return the precision and recall for each prediction.
    """
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    tp = 0
    fp = 0
    fn = len(ground_truth)  # Total number of ground truth objects for the current class
    tn = 0  # Assuming this will be calculated for each class for a binary task (more complex for multi-class)
    
    precisions = []
    recalls = []
    
    for prediction in predictions:
        pred_box = prediction['bbox']
        pred_class = prediction['category_id']
        matched = False
        
        # Iterate over ground truth to find a match
        for i, gt in enumerate(ground_truth):
            gt_box = gt['bbox']
            if not gt.get('matched', False) and calculate_iou(pred_box, gt_box) >= iou_threshold:
                ground_truth[i]['matched'] = True  # Mark this ground truth as matched
                matched = True
                fn -= 1  # Reduce FN when matched
                break
        
        if matched:
            tp += 1
        else:
            fp += 1
        
        # Calculate precision and recall at this point (after each prediction)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Append precision and recall for this prediction
        precisions.append(precision)
        recalls.append(recall)

    return tp, fp, fn, tn, precisions, recalls


def calculate_auc(precisions, recalls):
    """
    Calculate the Area Under the Precision-Recall Curve (AUC-PR) using the trapezoidal rule.
    """
    return np.trapz(precisions, recalls)  # Using the trapezoidal rule for integration


def save_class_precision_recall_curve_to_pdf(precisions, recalls, class_id, class_names, auc, output_folder):
    """
    Save the Precision-Recall curve for a single class to a PDF file in the specified folder.
    """
    class_name = class_names[class_id - 1]  # Adjust because class IDs start from 1 in the file
    
    # Ensure the output folder for individual class PR curves exists
    class_folder = os.path.join(output_folder, 'class_pr_curves')
    os.makedirs(class_folder, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'{class_name} (AUC={auc:.4f})', color='blue')
    plt.title(f'Precision-Recall Curve for {class_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot as a PDF using the class name in the 'class_pr_curves' folder
    output_pdf_path = os.path.join(class_folder, f'{class_name}_pr_curve.pdf')
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()


def save_precision_recall_curves_to_pdf(all_precisions, all_recalls, all_class_ids, auc_per_class, class_names, output_pdf_path):
    """
    Save the Precision-Recall curves for all classes to a PDF file.
    """
    plt.figure(figsize=(8, 6))

    # Define a list of colors for each class
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    for i, class_id in enumerate(all_class_ids):
        class_name = class_names[class_id - 1]  # Adjust because class IDs start from 1 in the file
        plt.plot(all_recalls[class_id], all_precisions[class_id], label=f'{class_name} (AUC={auc_per_class[class_id]:.4f})', color=colors[i % len(colors)])
    
    plt.title('Precision-Recall Curves for All Classes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot as a PDF
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    # Calculate the intersection area
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    intersection_area = inter_w * inter_h
    
    # Calculate the union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def load_json(file_path):
    # You can use this helper function to load the JSON data
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_precision_recall_for_all_classes(ground_truth_json, predictions_json, class_names_file, output_folder):
    # Load class names from the provided .txt file
    class_names = load_class_names(class_names_file)

    # Load ground truth and predictions from JSON files
    ground_truth = load_json(ground_truth_json)
    predictions = load_json(predictions_json)
    
    # Group ground truth data by class
    ground_truth_by_class = defaultdict(list)
    for item in ground_truth['annotations']:
        ground_truth_by_class[item['category_id']].append(item)
    
    # Dictionaries to hold precision, recall, and AUC values for all classes
    all_precisions = defaultdict(list)
    all_recalls = defaultdict(list)
    auc_per_class = {}
    confusion_matrix = {}
    all_class_ids = list(ground_truth_by_class.keys())

    # Calculate precision, recall for each class and store the results
    for class_id in ground_truth_by_class:
        class_name = class_names[class_id - 1]  # Get class name from the list (class_id starts at 1)
        class_ground_truth = ground_truth_by_class[class_id]
        class_predictions = [pred for pred in predictions if pred['category_id'] == class_id]
        
        # Calculate precision, recall, and confusion matrix for this class
        tp, fp, fn, tn, precisions, recalls = calculate_precision_recall_for_class(class_ground_truth, class_predictions)
        
        # Store the precision and recall for this class
        all_precisions[class_id] = precisions
        all_recalls[class_id] = recalls
        
        # Calculate AUC for this class and store it
        auc_per_class[class_id] = calculate_auc(precisions, recalls)

        # Store confusion matrix values
        confusion_matrix[class_id] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
        
        # Save the PR curve for this class to a separate PDF
        save_class_precision_recall_curve_to_pdf(precisions, recalls, class_id, class_names, auc_per_class[class_id], output_folder)

    # Save the PR curves for all classes to a single PDF
    save_precision_recall_curves_to_pdf(all_precisions, all_recalls, all_class_ids, auc_per_class, class_names, os.path.join(output_folder, 'all_classes_pr_curve.pdf'))
    
    # Save the confusion matrix, precision, recall, AUC, and mAP to a text file
    with open(os.path.join(output_folder, 'metrics_summary.txt'), 'w') as f:
        for class_id in confusion_matrix:
            class_name = class_names[class_id - 1]
            cm = confusion_matrix[class_id]
            precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
            recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
            f.write(f'Class: {class_name} (ID: {class_id})\n')
            f.write(f'Confusion Matrix: TP={cm["TP"]}, FP={cm["FP"]}, TN={cm["TN"]}, FN={cm["FN"]}\n')
            f.write(f'Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc_per_class[class_id]:.4f}\n\n')

        # Calculate and write the mAP (mean average precision)
        mAP = np.mean(list(auc_per_class.values()))
        f.write(f'mAP: {mAP:.4f}')

    # Return the AUC values for all classes
    return auc_per_class
