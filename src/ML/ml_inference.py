import numpy as np
import tensorflow as tf
import keras_cv

import cv2
import matplotlib.pyplot as plt

yolo_mdl_str = "yolo_v8_xs_backbone_coco"

obj_dl_resolution = [640, 640]
class_ids = ["Protein line", "RNA line", "Strip", "Control grid line", "Control line"]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


def load_model_h5_tf(cur_mdl, class_ids, h5_model_path):
    # Check if cur_mdl is None
    if cur_mdl is not None:
        print("Model is already loaded.")
        return cur_mdl
    # Load the model from the specified path

    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        yolo_mdl_str, load_weights=False
    )

    self_yolo_model = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_ids),
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=2,  # as recommended by Keras
    )

    self_yolo_model.load_weights(h5_model_path)

    return self_yolo_model


# Function to preprocess input image
# Load the image with all RGB channels
def preprocess_image_tf(image_path):
    # Define the resizing layer with padding to aspect ratio
    inference_resizer = keras_cv.layers.Resizing(
        obj_dl_resolution[0],
        obj_dl_resolution[1],
        bounding_box_format="xyxy",
        pad_to_aspect_ratio=True,
    )

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    original_shape = tf.shape(image)[:2]

    image = tf.cast(image, tf.float32)  # / 255.0
    image = inference_resizer({"images": tf.expand_dims(image, axis=0)})["images"]

    return image, original_shape  # Add batch dimension and return original shape
    # image = tf.image.resize_with_pad(image, target_height=640, target_width=640)
    # image = image / 255.0  # Normalize to [0, 1]


def infer_image_tf(cur_mdl, preprocessed_image):
    # Perform inference
    predictions = cur_mdl.predict(preprocessed_image)
    return predictions


# Function to postprocess model output
def postprocess_output_tf(output, original_shape, threshold=0.55):
    # Determine the scaling factor and padding
    original_height, original_width = (
        tf.cast(original_shape[0], dtype=tf.float32),
        tf.cast(original_shape[1], dtype=tf.float32),
    )

    scale = min(
        obj_dl_resolution[0] / original_height, obj_dl_resolution[1] / original_width
    )

    boxes, scores, classes = output["boxes"], output["confidence"], output["classes"]

    # print(boxes, scores, classes)
    # Apply threshold to filter out low confidence detections
    mask = scores > threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    # Also fiter out boxes where its x_max is larger
    # than the original image width buy a margin of 1 pixel

    x_max_mask = (
        boxes[:, 2] / scale < tf.cast(original_shape[1], dtype=tf.float32) + 5.0
    )
    boxes = boxes[x_max_mask]
    scores = scores[x_max_mask]
    classes = classes[x_max_mask]

    # Keep only the max score per class
    unique_classes = tf.unique(classes)[0]
    selected_boxes = []
    selected_scores = []
    selected_classes = []

    for cls in unique_classes:
        # Flatten to get 1D indices
        cls_mask = tf.where(classes == cls)[:, 0]
        # The extract scores for the current class, it will be a new tensor/list
        cls_scores = tf.gather(scores, cls_mask)

        max_score_index = tf.argmax(cls_scores)

        selected_idx = cls_mask[max_score_index]

        selected_boxes.append(boxes[selected_idx])
        selected_scores.append(scores[selected_idx])
        selected_classes.append(classes[selected_idx])

    boxes = tf.stack(selected_boxes)
    scores = tf.stack(selected_scores)
    classes = tf.stack(selected_classes)

    # pad_height = (obj_dl_resolution[0] - original_height * scale)
    # pad_width = (obj_dl_resolution[1] - original_width * scale)

    # Scale boxes back to original image shape
    boxes = tf.convert_to_tensor(boxes)
    boxes = tf.stack(
        [
            boxes[:, 0] / scale,
            boxes[:, 1] / scale,
            boxes[:, 2] / scale,
            boxes[:, 3] / scale,
        ],
        axis=-1,
    )

    return boxes, scores, classes


def class_name_to_id_tf(class_name):
    for class_id, name in class_mapping.items():
        if name == class_name:
            return class_id
    return None  # Return None if class name not found


def rtn_pos_xywh(boxes, classes, target_class_id):
    # Find indices where classes match the target class ID
    target_indices = tf.where(classes == target_class_id)[:, 0]

    # Extract boxes corresponding to the target class
    target_boxes = tf.gather(boxes, target_indices)

    # turn all coordinates to positive
    target_boxes = tf.maximum(target_boxes, 0)

    # Convert it to x, y, w, h format in numpy array
    target_boxes = tf.stack(
        [
            target_boxes[:, 0],
            target_boxes[:, 1],
            target_boxes[:, 2] - target_boxes[:, 0],
            target_boxes[:, 3] - target_boxes[:, 1],
        ],
        axis=-1,
    )

    return target_boxes.numpy()  # Convert to numpy array


def rtn_pos_xyxy(boxes, classes, target_class_id):
    # Find indices where classes match the target class ID
    target_indices = tf.where(classes == target_class_id)[:, 0]

    # Extract boxes corresponding to the target class
    target_boxes = tf.gather(boxes, target_indices)

    # if no boxes found, return empty array
    if tf.shape(target_boxes)[0] == 0:
        return np.array([])

    # turn all coordinates to positive
    target_boxes = tf.maximum(target_boxes, 0)

    # Convert it to x, y, w, h format in numpy array
    target_boxes = tf.stack(
        [
            target_boxes[:, 0],
            target_boxes[:, 1],
            target_boxes[:, 2],
            target_boxes[:, 3],
        ],
        axis=-1,
    )

    return target_boxes.numpy()  # Convert to numpy array


def get_color(class_id):
    if class_id == 0:
        return (0, 0, 255)  # red | Protein line (BGR)
    elif class_id == 1:
        return (0, 165, 255)  # orange | RNA line
    elif class_id == 2:
        return (0, 255, 255)  # yellow | Strip
    elif class_id == 3:
        return (0, 255, 0)  # green | Control grid line
    elif class_id == 4:
        return (255, 0, 0)  # blue | Control line
    else:
        return (0, 0, 0)


def display_detections_tf(cv_img, boxes, scores, classes, class_mapping):
    for box, img_class in zip(boxes, classes):
        # turn all coordinates to positive
        box = tf.maximum(box, 0)
        x_min, y_min, x_max, y_max = box.numpy().astype(int)

        class_name = class_mapping[img_class.numpy()]

        class_color = get_color(int(img_class))
        # Draw bounding box
        cv_img = cv2.rectangle(cv_img, (x_min, y_min), (x_max, y_max), class_color, 2)

        # Add class name text
        cv_img = cv2.putText(
            cv_img,
            class_name,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            class_color,
            2,
        )

    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
