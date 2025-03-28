from src.label_preprocessing import add_pose_tag_to_xml, convert_coordinates_to_int, convert_png_to_jpg, update_folder_tag, update_xml_files
from src.evaluate_model import convert_voc_to_coco, process_images, calculate_precision_recall_for_all_classes, process_images_parallel
from src.data_restructuring import delete_images_without_xml, update_paths, copy_all_files
import os
import random
import shutil
import time


from tflite_model_maker.config import ExportFormat
from tflite_model_maker import object_detector
from absl import logging

import tensorflow as tf
import numpy as np
from tensorboard.plugins.hparams import api as hp


assert tf.__version__.startswith("2")


# Define Hyperparameters
HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(20, 100))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-5, 1e-2))

hparams = [HP_EPOCHS, HP_BATCH_SIZE, HP_LEARNING_RATE]

tf.get_logger().setLevel("ERROR")

logging.set_verbosity(logging.ERROR)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))


use_custom_dataset = True  # @param ["False", "True"] {type:"raw"}

dataset_is_split = False  # @param ["False", "True"] {type:"raw"}

label_map = {
    1: "car",
    2: "pedestrian",
    3: "stop-sign",
    4: "crosswalk-sign",
    5: "parking-sign",
    6: "thirty-sign",
    7: "no-thirty-sign",
    8: "go-left-sign",
    9: "go-right-sign",
    10: "intersection-right-of-way-sign",
    11: "intersection-grant-sign",
    12: "expressway-start",
    13: "expressway-end",
    14: "barred-area",
    15: "pedestrian-island-right",
    16: "no-passing-start",
    17: "no-passing-end",
}

if dataset_is_split:
    # If your dataset is already split, specify each path:
    train_images_dir = "/workspace/dataset/train/images"
    train_annotations_dir = "/workspace/dataset/train/annotations"
    val_images_dir = "/workspace/dataset/validation/images"
    val_annotations_dir = "/workspace/dataset/validation/annotations"
    test_images_dir = "/workspace/dataset/test/images"
    test_annotations_dir = "/workspace/dataset/test/annotations"
else:
    # If it's NOT split yet, specify the path to all images and annotations
    images_in = '/workspace/dataset/images'
    annotations_in = '/workspace/dataset/annotations'


# files to evaluate the Ground Truth of the Test-dataset (saved in a single .json)
voc_annotations_path = '/workspace/split-dataset/test/annotations'
image_folder_path = '/workspace/split-dataset/test/images'
class_file = '/workspace/sign-labels.txt'  # Path to the .txt file containing class names

# folder to output the results
output_folder = '/output'  # Main output folder


# Split Dataset i train, test and validation
def split_dataset(images_path, annotations_path, test_split=0.2, val_split=0, out_path='split-dataset'):
    """Splits a directory of sorted images/annotations into training, validation, and test sets.
    
    By default it is expected to have a default high quality validation set, 
    to get best results in training

    Args:
      images_path: Path to the directory with your images (JPGs).
      annotations_path: Path to a directory with your VOC XML annotation files,
        with filenames corresponding to image filenames. This may be the same path
        used for images_path.
      val_split: Fraction of data to reserve for validation (float between 0 and 1).
      test_split: Fraction of data to reserve for test (float between 0 and 1).

    Returns:
      The paths for the split images/annotations (train_dir, val_dir, test_dir)
    """

    _, dirs, _ = next(os.walk(images_path))

    # Define output directories
    train_dir = os.path.join(out_path, "train")
    val_dir = os.path.join(out_path, "validation")
    test_dir = os.path.join(out_path, "test")
    
    # only do the splitting if there is no split-data dir yet
    if os.path.exists(train_dir) and os.path.isdir(train_dir) and not os.listdir(train_dir):
        return (train_dir, val_dir, test_dir)
    else:
        """
            We want a constant validation set to allow best results when 
            trying to evaluate which is the best model.
            For first testing we still want to keep a default split though
            Thats why we do check here, whether a validation set already exists
            
            It is expected that the val_dir is correctly structured 
            (annotations + images folders) containing the data
        
        """
        
        # Create subdirectories for images and annotations
        IMAGES_TRAIN_DIR = os.path.join(train_dir, "images")
        IMAGES_VAL_DIR = os.path.join(val_dir, "images")
        IMAGES_TEST_DIR = os.path.join(test_dir, "images")
        os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
        os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
        os.makedirs(IMAGES_TEST_DIR, exist_ok=True)

        ANNOT_TRAIN_DIR = os.path.join(train_dir, "annotations")
        ANNOT_VAL_DIR = os.path.join(val_dir, "annotations")
        ANNOT_TEST_DIR = os.path.join(test_dir, "annotations")
        os.makedirs(ANNOT_TRAIN_DIR, exist_ok=True)
        os.makedirs(ANNOT_VAL_DIR, exist_ok=True)
        os.makedirs(ANNOT_TEST_DIR, exist_ok=True)

        # Get all filenames for this dir, filtered by filetype
        filenames = os.listdir(os.path.join(images_path))
        filenames = [
            os.path.join(images_path, f) for f in filenames if (f.endswith(".jpg"))
        ]

        # Shuffle the files deterministically
        filenames.sort()
        random.seed(42)
        random.shuffle(filenames)

        # Get exact number of images for validation and test; the rest is for training
        val_count = int(len(filenames) * val_split)
        test_count = int(len(filenames) * test_split)

        for i, file in enumerate(filenames):
            source_dir, filename = os.path.split(file)
            annot_file = os.path.join(annotations_path, filename.replace("jpg", "xml"))

            if i < val_count:
                shutil.copy(file, IMAGES_VAL_DIR)
                shutil.copy(annot_file, ANNOT_VAL_DIR)
                update_folder_tag(
                    os.path.join(ANNOT_VAL_DIR, filename.replace("jpg", "xml")),
                    "/workspace/split-dataset/validation/images",
                )  # Update folder tag
            elif i < val_count + test_count:
                shutil.copy(file, IMAGES_TEST_DIR)
                shutil.copy(annot_file, ANNOT_TEST_DIR)
                update_folder_tag(
                    os.path.join(ANNOT_TEST_DIR, filename.replace("jpg", "xml")),
                    "/workspace/split-dataset/test/images",
                )  # Update folder tag
            else:
                shutil.copy(file, IMAGES_TRAIN_DIR)
                shutil.copy(annot_file, ANNOT_TRAIN_DIR)
                update_folder_tag(
                    os.path.join(ANNOT_TRAIN_DIR, filename.replace("jpg", "xml")),
                    "/workspace/split-dataset/train/images",
                )  # Update folder tag

    return (train_dir, val_dir, test_dir)


def custom_evaluation(tflite_model_filename, test_data, labels_filename, image_folder_path, output_folder):
    # Convert VOC annotations to COCO format
    convert_voc_to_coco(voc_annotations_path, os.path.join(output_folder, image_folder_path), os.path.join(output_folder, labels_filename), output_folder)
    
    # Run the inference process on the test data using the TFLite model
    process_images_parallel(
        model_filename=tflite_model_filename, 
        labels_filename=labels_filename, 
        images_path=image_folder_path, 
        output_folder=output_folder,
        threshold=0.5,
        num_workers=8,
    )
    
    # Calculate precision, recall, and mAP
    auc_per_class = calculate_precision_recall_for_all_classes(os.path.join(output_folder, 'annotations_coco.json'), os.path.join(output_folder, 'predictions_coco.json'), os.path.join(output_folder, labels_filename), os.path.join(output_folder, 'results'))
    print(f"Per-class AUC: {auc_per_class}")
    
    mAP = np.mean(list(auc_per_class.values()))
    print(f"mAP: {mAP:.4f}")


# Function to create a subfolder for each hyperparameter combination
def create_hyperparameter_dir(hparams_dict):
    param_str = f"epochs-{hparams_dict[HP_EPOCHS]}_batch_size-{hparams_dict[HP_BATCH_SIZE]}_learning_rate-{hparams_dict[HP_LEARNING_RATE]}"
    dir_path = os.path.join(output_folder, param_str)
    return dir_path

# Function to check if the directory exists
def check_and_create_dir(hparams_dict):
    dir_path = create_hyperparameter_dir(hparams_dict)
    if not os.path.exists(dir_path):  # Check if the folder already exists
        os.makedirs(dir_path)  # Create the folder if it doesn't exist
        return False  # Return False to indicate we should train the model
    else:
        return True  # Return True to indicate the folder already exists, no need to train


def modify_model_spec(learning_rate):
    # Assuming EfficientDetLite0Spec
    spec = object_detector.EfficientDetLite0Spec()
    
    # Modify the learning rate directly in the spec if it's supported
    spec.learning_rate = learning_rate
    
    return spec


# Function to train the model and evaluate using the custom evaluation logic
def train_and_evaluate(hparams_dict, train_data, validation_data, test_data, image_folder_path):
    print("TF is using:", tf.config.experimental.list_logical_devices('GPU'))
    if check_and_create_dir(hparams_dict):
        print(f"Skipping training for hyperparameters: {hparams_dict}")
        return  # Skip training if the directory already exists
    
    model_dir = create_hyperparameter_dir(hparams_dict)
    learning_rate = hparams_dict[HP_LEARNING_RATE]
    
    # Modify model_spec with the learning rate
    spec = modify_model_spec(learning_rate)
    # Create the model using specified hyperparameters
    model = object_detector.create(
        train_data=train_data,
        model_spec=modify_model_spec(hparams_dict[HP_LEARNING_RATE]),
        validation_data=validation_data,
        epochs=hparams_dict[HP_EPOCHS],
        batch_size=hparams_dict[HP_BATCH_SIZE],
        train_whole_model=True,
    )

    # # Export the model to the created directory
    tflite_filename = os.path.join(model_dir, "model.tflite")
    labels_filename = os.path.join(model_dir, "labels.txt")
    
    model.export(
        export_dir=model_dir,
        tflite_filename=tflite_filename,
        label_filename=labels_filename,
        export_format=[ExportFormat.TFLITE, ExportFormat.LABEL],
    )

    # Use the custom evaluation function
    custom_evaluation(tflite_filename, test_data, labels_filename, image_folder_path, model_dir)


if __name__ == '__main__':
    train_dir = '/workspace/split-dataset/train'
    dataset_image_dir = "/workspace/dataset/images"
    
    # Prepare labels for training (adjust paths & names)
    base_image_folder = "/workspace/training_data/images"
    base_xml_folder = "/workspace/training_data/labels"
    update_paths(base_xml_folder, "/workspace/dataset/images")
    delete_images_without_xml(base_image_folder, base_xml_folder)

    # Copy images to the dataset folder
    src_folder = '/workspace/training_data/images'
    dest_folder = '/workspace/dataset/images'
    copy_all_files(src_folder, dest_folder)

    # Copy labels to the dataset folder
    src_folder = '/workspace/training_data/labels'
    dest_folder = '/workspace/dataset/annotations'
    copy_all_files(src_folder, dest_folder)



    # Call the function to convert PNG to JPG
    start_time = time.time()
    convert_png_to_jpg(images_in)
    end_time = time.time()
    print(f"convert_png_to_jpg took {end_time - start_time:.2f} seconds")

    # Update Path Property in Annotations
    start_time = time.time()
    update_xml_files(annotations_in, images_in)
    end_time = time.time()
    print(f"update_xml_files took {end_time - start_time:.2f} seconds")

    # Add Pose Tag in Annotation (needed to match the Pascal VOC format)
    start_time = time.time()
    add_pose_tag_to_xml(annotations_in)
    end_time = time.time()
    print(f"add_pose_tag_to_xml took {end_time - start_time:.2f} seconds")

    # Convert Coordinates in Annotations from floats to ints
    start_time = time.time()
    convert_coordinates_to_int(annotations_in)
    end_time = time.time()
    print(f"convert_coordinates_to_int took {end_time - start_time:.2f} seconds")

    
    # We need to instantiate a separate DataLoader for each split dataset
    if dataset_is_split:
        train_data = object_detector.DataLoader.from_pascal_voc(
            train_images_dir, train_annotations_dir, label_map=label_map
        )
        validation_data = object_detector.DataLoader.from_pascal_voc(
            val_images_dir, val_annotations_dir, label_map=label_map
        )
        test_data = object_detector.DataLoader.from_pascal_voc(
            test_images_dir, test_annotations_dir, label_map=label_map
        )
    else:
        print('Dataset not split')
        train_dir, val_dir, test_dir = split_dataset(
            images_in,
            annotations_in,
            test_split=0.2,
            val_split=0.2,
            out_path="/workspace/split-dataset",
        )


        print(label_map)
        print('dataset split')
        train_data = object_detector.DataLoader.from_pascal_voc(
            os.path.join(train_dir, "images"),
            os.path.join(train_dir, "annotations"),
            label_map=label_map,
        )
        validation_data = object_detector.DataLoader.from_pascal_voc(
            os.path.join(val_dir, "images"),
            os.path.join(val_dir, "annotations"),
            label_map=label_map,
        )
        test_data = object_detector.DataLoader.from_pascal_voc(
            os.path.join(test_dir, "images"),
            os.path.join(test_dir, "annotations"),
            label_map=label_map,
        )

    print(f"train count: {len(train_data)}")
    print(f"validation count: {len(validation_data)}")
    print(f"test count: {len(test_data)}")


    spec = object_detector.EfficientDetLite0Spec()
    for epochs in range(HP_EPOCHS.domain.min_value, HP_EPOCHS.domain.max_value + 1, 10):  # Step size 10
        for batch_size in HP_BATCH_SIZE.domain.values:  # Correct way to iterate over Discrete values
            for lr in np.linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, num=5):  # 5 values for learning rate
                hparams_dict = {
                    HP_EPOCHS: epochs,
                    HP_BATCH_SIZE: batch_size,
                    HP_LEARNING_RATE: lr,
                }
                train_and_evaluate(hparams_dict, train_data, validation_data, test_data, image_folder_path)
    # for epochs in HP_EPOCHS.domain.values:  # Single value: [20]
    #     for batch_size in HP_BATCH_SIZE.domain.values:  # Single value: [16]
    #         for lr in HP_LEARNING_RATE.domain.values:  # Iterate over the given discrete learning rates
    #             hparams_dict = {
    #                 HP_EPOCHS: epochs,
    #                 HP_BATCH_SIZE: batch_size,
    #                 HP_LEARNING_RATE: lr,
    #             }
    #             train_and_evaluate(hparams_dict, train_data, validation_data, test_data, image_folder_path)