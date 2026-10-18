from dataclasses import dataclass
from skimage import io, color

import skimage

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve

import cv2
import os
import sys
import csv

from typing import Any

# from ..ML import ml_inference  # type: ignore[misc]
from ML import ml_inference  # type: ignore[misc]


@dataclass
class DetectionResults:
    band: dict[int, Any]
    grid: list

    ml_box: tf.Tensor | None = None
    ml_score: tf.Tensor | None = (
        None  # detection confidences per object (when using YOLO)
    )
    ml_class: tf.Tensor | None = None  # detected class IDs per object (when using YOLO)

    band_pos_profile: dict[int, Any] | None = None  # optional band profile data
    band_baseline: dict[int, Any] | None = None  # optional band baseline data
    band_corrected: dict[int, Any] | None = (
        None  # optional band corrected profile data        #
    )
    band_corrected_lbs: dict[int, Any] | None = None  # left bases of detected peaks
    band_corrected_rbs: dict[int, Any] | None = None  # right bases of detected peaks

    band_corrected_peak_areas: dict[int, Any] | None = (
        None  # peak areas of detected peaks
    )

    relaxed_peak_detection_used: bool = (
        False  # whether to use relaxed peak detection for bands
    )

    grid_pos_profile: list | None = None  # optional grid profile data
    grid_baseline: list | None = None  # optional grid baseline data
    grid_corrected: list | None = None  # optional grid corrected profile data
    grid_corrected_lbs: list | None = None  # left bases of detected peaks
    grid_corrected_rbs: list | None = None  # right bases of detected peaks
    grid_corrected_peak_areas: list | None = None  # peak areas of detected peaks


class BasicStripWithGrid:
    # These are the expected ratio ranges for bands and grid positions,
    # based on the standard strip size and layout
    BAND0_RATIO_RANGE = (0.3, 0.52)
    BAND1_RATIO_RANGE = (0.5, 0.72)
    BAND_HEIGHT_RATIO_RANGE = (0.1, 0.15)

    BAND0_EST_Y_RATIO = 0.35
    BAND1_EST_Y_RATIO = 0.55

    LANE_THRESHOLD_FRAC = 0.25
    # These are the expected distance ratios from bands to grid
    # prefer using grid to find bands for better accuracy
    GRID_TO_END_MARGIN_RATIO = 0.2  # grid to bottom margin ratio
    BAND0_GRID_DIST_RATIO_RANGE = (
        20,
        27,
    )  # band0 to grid y-diff ratio , top to bottom
    BAND1_GRID_DIST_RATIO_RANGE = (
        10,
        17,
    )  # band1 to grid y-diff ratio , top to bottom
    BAND_GRID_HEIGHT_RATIO_RANGE = (4.0, 6.0)

    # ** Another set of relaxed ratios that are used when the strict ratios fail **
    LANE_THRESHOLD_FRAC_RELAXED = 0.20
    BAND0_GRID_DIST_RATIO_RANGE_RELAXED = (
        19,
        27,
    )  # band0 to grid y-diff ratio , top to bottom
    BAND1_GRID_DIST_RATIO_RANGE_RELAXED = (
        10,
        18,
    )  # band1 to grid y-diff ratio , top to bottom
    BAND_GRID_HEIGHT_RATIO_RANGE_RELAXED = (4.0, 7)

    # The distance between bands for ML detection, divided by grid height
    ML_BAND_DIST_RATIO_RANGE = (12, 13.8)

    def __init__(
        self,
        file_path: str,
        concentration: dict,
        band_cnts: int = 2,
        ml_mdl: Any = None,
        **kwargs,
    ):
        self.file_path = file_path
        self.concentration = concentration

        self.ml_mdl = ml_mdl

        self.img_skimage = None
        self.img_opencv = None

        self.band_cnts = band_cnts

        self.cropped_strip = None

        self.ml_detection_results = None
        self.manual_detection_results = None

        self.final_detection_results = None
        self.used_ml = False

        # Only for fallback method
        self.morph_kernel_size = kwargs.get("morph_kernel_size", (3, 60))
        self.mask_base = None
        self.mask_morphology = None
        self.mask_filtered = None

    def load_img(self):
        """
        Load image using skimage and OpenCV
        Convert to grayscale float image for skimage
        0..1 float for skimage, 0..255 uint8 for OpenCV
        """

        img_skimage = None

        img_skimage = io.imread(self.file_path)
        if img_skimage.shape[2] == 4:
            img_skimage = color.rgba2rgb(img_skimage)  # convert RGBA -> RGB

        # We will operate in floating point range [0,1]
        img_skimage = skimage.img_as_float(img_skimage)

        if img_skimage.ndim == 3:
            # img_gray = img_as_float(color.rgb2gray(img))
            # img_skimage = skimage.img_as_float(img_skimage.mean(axis=2))
            img_skimage = img_skimage.mean(axis=2)
        # else:
        #    img_skimage = skimage.img_as_float(img_skimage)

        # print(img_skimage)

        self.img_skimage = img_skimage
        self.img_opencv = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)

        print(f"Image loaded: {self.file_path} with shape {self.img_skimage.shape}")

    def conclude_detection_results(self):
        """
        Conclude the final detection results by prioritizing ML-based detection
        results if available; otherwise, use manual detection results.
        """
        if self.used_ml is True and self.ml_detection_results is not None:
            self.final_detection_results = self.ml_detection_results

            # extract missing bands from manual detection if any,
            # Don't need to extract grid and strip because ML should always detect them
            # But just in case, we check and give warning
            for band_id in range(self.band_cnts):
                if band_id not in self.final_detection_results.band:
                    # raise error, cannot find band in both detections
                    raise ValueError(
                        "Error: "
                        + os.path.basename(self.file_path)
                        + f" --- Band {band_id} missing in ML "
                        + "detection. Write your code better to enable "
                        + "manual detection to fill in missing bands."
                    )

            if len(self.final_detection_results.grid) == 0:
                raise ValueError(
                    "Error: "
                    + os.path.basename(self.file_path)
                    + " --- Grid missing in ML detection. Run manual detection "
                    + "to get the grid locations."
                )

            print(f"Using ML detection results for {os.path.basename(self.file_path)}")

        else:
            if self.manual_detection_results is None:
                raise ValueError(
                    "Error: "
                    + os.path.basename(self.file_path)
                    + " --- No detection results available from ML or manual methods."
                )

            # use manual detection results, it should have all bands/grid/strip
            self.final_detection_results = self.manual_detection_results

            print(
                os.path.basename(self.file_path)
                + " --- Using manual detection "
                + "results as final detection results."
            )

        # initialize empty profile
        self.final_detection_results.band_pos_profile = {}
        self.final_detection_results.grid_pos_profile = []

    # Methods that uses ml_inference.py for ML detection
    def strip_detect_crop_ml(self, threshold=0.4, debug=False):
        """
        ML-based strip detection method using the provided ML model.
        Detects the strip region in the image, crops it, and stores the cropped
        strip image.

        Raises:
            ValueError: If no strip is detected in the image or ML model is not provided.
        """

        if self.ml_mdl is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + "\n ML model not provided. Cannot perform ML-based detection."
            )

        # Load and preprocess image
        preprocessed_image, original_shape = ml_inference.preprocess_image_tf(
            self.file_path
        )

        # Perform inference
        predictions = ml_inference.infer_image_tf(self.ml_mdl, preprocessed_image)

        # Postprocess the predictions
        boxes, scores, classes = ml_inference.postprocess_output_tf(
            predictions, original_shape, threshold=threshold
        )

        if debug:
            # Display detected boxes on the image
            ml_inference.display_detections_tf(
                self.img_opencv, boxes, scores, classes, ml_inference.class_mapping
            )

        self.ml_detection_results = DetectionResults(
            band={},
            grid=[],
            ml_box=boxes,
            ml_score=scores,
            ml_class=classes,
        )

        # Find the strip box, and crop
        # The x corrdinates for the rest of the objects will be based on the strip box
        self.strip_crop_from_detections_ml(boxes, classes)

    def strip_crop_from_detections_ml(self, boxes, classes):
        strip_class_id = ml_inference.class_name_to_id_tf("Strip")

        if strip_class_id is None:
            raise ValueError("Error: Strip class not found in class mapping.")
        strip_boxes = ml_inference.rtn_pos_xyxy(boxes, classes, strip_class_id)
        print(strip_boxes)

        if len(strip_boxes) == 0:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- No strip detected in ML-based detection."
            )

        # crop the entire image using the x left and x right coordinates of the strip box
        strip_box = strip_boxes[0]  # x1, y1, x2, y2

        x1 = int(strip_box[0])
        x2 = int(strip_box[2])

        if self.img_skimage is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Image not loaded. Call load_img() before detection."
            )

        self.cropped_strip = self.img_skimage[:, x1:x2]

    def extract_1grid_ml(self, ml_pad=5, debug=False):
        H, W = self.cropped_strip.shape

        grid_class_id = ml_inference.class_name_to_id_tf("Control grid line")

        if grid_class_id is None:
            raise ValueError(
                "Error: Control grid line class not found in class mapping."
            )

        boxes, _, classes = (
            self.ml_detection_results.ml_box,
            self.ml_detection_results.ml_score,
            self.ml_detection_results.ml_class,
        )
        grid_boxes = ml_inference.rtn_pos_xywh(boxes, classes, grid_class_id)
        print(grid_boxes)

        if len(grid_boxes) == 0:
            # If AI cannot find grid, probably there is no grid
            # So just raise Error
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- No grid detected in ML-based detection."
            )

        # For grid, we only need the y and h
        grid_box = grid_boxes[0]  # x, y, w, h

        x = 0
        y = int(grid_box[1])
        h = int(grid_box[3])

        self.ml_detection_results.grid = [(x, y - ml_pad, W, h + 2 * ml_pad)]

        print(
            os.path.basename(self.file_path)
            + f" --- Detected grid at {self.ml_detection_results.grid[0]} using ML."
        )

        if debug:
            # display the rect on the cropped strip
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.imshow(self.cropped_strip, cmap="gray")
            rect = self.ml_detection_results.grid[0]
            x, y, w, h = rect
            rect_patch = plt.Rectangle(
                (x, y), w, h, edgecolor="green", facecolor="none", linewidth=2
            )
            ax.add_patch(rect_patch)
            plt.title("ML-detected Grid")
            plt.axis("on")
            plt.show()

    def extract_2bands_ml(self, ml_pad=5, estimation_pad=15, debug=False):
        # Check if cropped strip is available
        if self.cropped_strip is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )

        if (
            self.ml_detection_results.grid is None
            or len(self.ml_detection_results.grid) == 0
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Grid position not available. Perform grid detection first."
            )

        H, W = self.cropped_strip.shape

        protein_class_id = ml_inference.class_name_to_id_tf("Protein line")
        RNA_class_id = ml_inference.class_name_to_id_tf("RNA line")

        if protein_class_id is None or RNA_class_id is None:
            print(
                "Warning: "
                + os.path.basename(self.file_path)
                + " --- Protein line or RNA line class not found in class mapping."
            )
            return False

        boxes, _, classes = (
            self.ml_detection_results.ml_box,
            self.ml_detection_results.ml_score,
            self.ml_detection_results.ml_class,
        )

        RNA_boxes = ml_inference.rtn_pos_xywh(boxes, classes, RNA_class_id)
        protein_boxes = ml_inference.rtn_pos_xywh(boxes, classes, protein_class_id)

        # Return False, and the main program should handle it by using manual detection
        if len(protein_boxes) == 0 and len(RNA_boxes) == 0:
            print(
                "Warning: "
                + os.path.basename(self.file_path)
                + " --- No bands detected in ML-based detection."
            )
            return False

        x = 0
        if RNA_boxes.size > 0:
            RNA_box = RNA_boxes[0]  # x, y, w, h
            RNA_y = int(RNA_box[1]) - ml_pad
            RNA_h = int(RNA_box[3]) + 2 * ml_pad

            self.ml_detection_results.band[0] = (x, RNA_y, W, RNA_h)

        if protein_boxes.size > 0:
            protein_box = protein_boxes[0]  # x, y, w, h
            protein_y = int(protein_box[1]) - ml_pad
            protein_h = int(protein_box[3]) + 2 * ml_pad

            self.ml_detection_results.band[1] = (x, protein_y, W, protein_h)

        print("Protein boxes:", protein_boxes)
        print("RNA boxes:", RNA_boxes)

        # Now we check if any band is missing, we will use estimation based on grid

        # Cannot use grid rect directly because ML box may have padding
        grid_class_id = ml_inference.class_name_to_id_tf("Control grid line")

        boxes, _, classes = (
            self.ml_detection_results.ml_box,
            self.ml_detection_results.ml_score,
            self.ml_detection_results.ml_class,
        )
        grid_boxes = ml_inference.rtn_pos_xywh(boxes, classes, grid_class_id)
        grid_h = grid_boxes[0][3]

        ml_band_dist_ratio_range = self.ML_BAND_DIST_RATIO_RANGE
        ml_band_dist_ratio_avg = (
            ml_band_dist_ratio_range[1] + ml_band_dist_ratio_range[0]
        ) / 2.0

        # RNA missing
        if 0 not in self.ml_detection_results.band:
            # est_RNA_y = int(
            #    protein_y - ml_band_dist_ratio_range[1] * grid_h - estimation_pad
            # )

            est_RNA_y = int(
                protein_y - ml_band_dist_ratio_avg * grid_h - estimation_pad
            )

            est_RNA_h = int(protein_h) + estimation_pad * 2

            self.ml_detection_results.band[0] = (x, est_RNA_y, W, est_RNA_h)

            print(
                os.path.basename(self.file_path)
                + f" --- Estimated RNA band at {self.ml_detection_results.band[0]} using ML estimation."
            )

        # Protein missing
        if 1 not in self.ml_detection_results.band:
            # est_protein_y = int(
            #    RNA_y + ml_band_dist_ratio_range[0] * grid_h - estimation_pad
            # )

            est_protein_y = int(
                RNA_y + ml_band_dist_ratio_avg * grid_h - estimation_pad
            )

            est_protein_h = int(RNA_h) + estimation_pad * 2

            self.ml_detection_results.band[1] = (x, est_protein_y, W, est_protein_h)

            print(
                os.path.basename(self.file_path)
                + f" --- Estimated Protein band at {self.ml_detection_results.band[1]} using ML estimation."
            )

        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

            # ---- Left: cropped strip with detections ----
            ax1.imshow(self.cropped_strip, cmap="gray")

            for band_id, rects in self.ml_detection_results.band.items():
                if band_id == 0:
                    band_color = (1.0, 0.647, 0.0)  # orange
                else:
                    band_color = (1.0, 0.0, 0.0)  # red

                x, y, w, h = rects
                rect_patch = plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor=band_color,
                    facecolor="none",
                    linewidth=2,
                )
                ax1.add_patch(rect_patch)

            ax1.set_title("ML-detected Bands")
            ax1.axis("on")

            # ---- Right: original image ----
            ax2.imshow(self.img_skimage, cmap="gray")
            ax2.set_title("Original Image")
            ax2.axis("on")

            plt.tight_layout()
            plt.show()

        # if debug:
        #     # display the rect on the cropped strip
        #     fig, ax = plt.subplots(figsize=(8, 10))
        #     ax.imshow(self.cropped_strip, cmap="gray")
        #     for band_id, rects in self.ml_detection_results.band.items():
        #         if band_id == 0:
        #             band_color = (1.0, 0.647, 0.0)  # orange
        #         else:
        #             band_color = (1.0, 0.0, 0.0)  # red

        #         for rect in rects:
        #             x, y, w, h = rect
        #             rect_patch = plt.Rectangle(
        #                 (x, y),
        #                 w,
        #                 h,
        #                 edgecolor=band_color,
        #                 facecolor="none",
        #                 linewidth=2,
        #             )
        #             ax.add_patch(rect_patch)

        #     plt.title("ML-detected Bands")
        #     plt.axis("on")
        #     plt.show()

        self.used_ml = True
        return True  # For now

    def _keep_best_component_fallback(self, mask, min_area=500, prefer_tall=True):
        """
        Private method: Keep the best connected component based on area
        and aspect ratio preference (keep largest / tall components).

        Args:
            mask (np.ndarray): Binary mask (0/255 uint8) of the detected regions.
            min_area (int): Minimum area threshold to consider a component.
            prefer_tall (bool): If True, prefer taller components in scoring.

        Returns:
            np.ndarray: Filtered binary mask with only the best component retained.
        """

        #  Connected component filtering (keep largest / tall components)
        # mask expected 0/255 uint8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if num_labels <= 1:
            return mask.copy()  # nothing found

        # stats columns: [x, y, w, h, area]
        best_label = None
        best_score = -1
        for lab in range(1, num_labels):
            x, y, w, h, area = stats[lab]
            if area < min_area:
                continue
            # compute score: area * aspect_ratio_penalty (favor tall)
            aspect = h / (w + 1e-6)

            # print(area, aspect)
            # preference to tall & large area
            score = area * (aspect if prefer_tall else 1.0)
            if score > best_score:
                best_score = score
                best_label = lab

        if best_label is None:
            # fallback: largest area (exclude background)
            areas = stats[1:, 4]
            best_label = np.argmax(areas) + 1

        out = np.zeros_like(mask)
        out[labels == best_label] = 255
        return out

    def strip_detect_crop_fallback(self):
        """
        Fallback strip detection method using traditional image processing.
        Detects the strip region in the image, crops it, and stores the cropped
        strip image.
        1. Improve contrast using histogram equalization.
        2. Apply Otsu's thresholding to create a binary mask.
        3. Perform morphological opening to remove small artifacts.
        4. Keep the best connected component based on area and aspect ratio.
        5. Crop the original image to the width of bounding box of the detected
        strip, retaining full height.

        Raises:
            ValueError: If no strip is detected in the image.
        """

        if self.img_opencv is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Image not loaded. Call load_img() before detection."
            )

        # Improve contrast
        img_eq = cv2.equalizeHist(self.img_opencv)

        # --- initial threshold (Otsu) -> baseline mask
        _, self.mask_base = cv2.threshold(
            img_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # ensure uint8
        self.mask_base = self.mask_base.astype(np.uint8)

        # --- morphological opening to remove small artifacts
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        self.mask_morphology = cv2.morphologyEx(
            self.mask_base, cv2.MORPH_OPEN, morph_kernel
        )

        self.mask_filtered = self._keep_best_component_fallback(
            self.mask_morphology, min_area=500, prefer_tall=True
        )

        contours, _ = cv2.findContours(
            self.mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- No strip detected in fallback method."
            )

        # pick the largest contour by area
        best_cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(best_cnt)

        # The cropping for the y will always be full height to retain grid lines
        # Cropping in done on img_skimage
        self.cropped_strip = self.img_skimage[:, x : x + w]

        # crop = img_eq[y:y+h, x:x+w]

        self.manual_detection_results = DetectionResults(
            band={},
            grid=[],
        )

    def display_masks_fallback(self):
        """
        Display the intermediate masks generated during the fallback strip detection process.
        Shows the original image, base mask (Otsu), mask after morphological operations,
        and the filtered mask with the best connected component overlaid on the original image.

        Raises:
            ValueError: If masks have not been computed. Call strip_detection_fallback() first.
        """

        if (
            self.mask_base is None
            or self.mask_morphology is None
            or self.mask_filtered is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Masks not computed. Call strip_detection_fallback() before displaying masks."
            )

        def overlay_contour_on_img(img_gray, mask, color=(255, 0, 0)):
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img_bgr, contours, -1, color, 2)
            return img_bgr[..., ::-1]  # RGB for plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(self.img_skimage, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(self.mask_base, cmap="gray")
        axes[1].set_title("Base Mask (Otsu)")
        axes[1].axis("off")

        axes[2].imshow(self.mask_morphology, cmap="gray")
        axes[2].set_title("After Morphology")
        axes[2].axis("off")

        # Show overlays
        overlay_imgs = overlay_contour_on_img(self.img_opencv, self.mask_filtered)

        axes[3].imshow(
            overlay_imgs,
        )
        axes[3].set_title("Filtered Mask (Best Component)")
        axes[3].axis("off")

        plt.show()

    def display_cropped_strip(self):
        """
        Display the cropped strip image, which is of the full height of the
        original image but cropped to the width of the detected strip.

        Raises:
            ValueError: If cropped strip is not available. Perform strip detection first.
        """

        if not hasattr(self, "cropped_strip"):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )

        print(
            os.path.basename(self.file_path)
            + " Displaying cropped strip image with shape:",
            self.cropped_strip.shape,
        )

        plt.figure(figsize=(8, 6))
        plt.imshow(self.cropped_strip, cmap="gray")
        plt.title("Cropped Strip Image")
        plt.axis("off")
        plt.show()

    def _auto_detect_lanes(
        self, img_gray, min_lane_size=5, threshold_frac=0.1, pad=4, debug=False
    ):
        """
        Private method: Automatically detect lane regions in a grayscale image
        using horizontal projection profiling.

        Args:
            img_gray (np.ndarray): Grayscale image (2D array) with pixel values in [0, 1].
            min_lane_size (int): Minimum height of a lane to be considered valid.
            threshold_frac (float): Fractional threshold for lane detection based
                                    on normalized projection.
            pad (int): Padding to apply to detected lane regions.
            debug (bool): If True, print debug information.
            Returns:
            list: List of detected lane rectangles in (x, y, w, h) format.
        """

        H, W = img_gray.shape
        length = H

        horizontal_proj = 1.0 - img_gray
        horizontal_proj = np.sum(horizontal_proj, axis=1)  # length W

        # normalize to 0..1
        horizontal_proj_norm = np.zeros_like(horizontal_proj)
        if horizontal_proj.max() - horizontal_proj.min() == 0:
            horizontal_proj_norm = (horizontal_proj - horizontal_proj.min()) / (
                horizontal_proj.max() - horizontal_proj.min() + 1e-12
            )
        else:
            horizontal_proj_norm = (horizontal_proj - horizontal_proj.min()) / (
                horizontal_proj.max() - horizontal_proj.min()
            )

        mask = horizontal_proj_norm > threshold_frac

        rects = []
        in_region = False

        start = 0

        for i in range(length):
            if mask[i] and not in_region:
                in_region = True
                start = i
            elif not mask[i] and in_region:
                end = i
                size = end - start

                if size >= min_lane_size:
                    y = max(0, start - pad)
                    h = min(H - y, size + 2 * pad)
                    rects.append((0, y, W, h))

                in_region = False

        # Handle region that continues to end
        if in_region:
            end = length
            size = end - start

            if size >= min_lane_size:
                y = max(0, start - pad)
                h = min(H - y, size + 2 * pad)
                rects.append((0, y, W, h))

        # np.set_printoptions(precision=4, suppress=True)
        # print(horizontal_proj_norm)
        # print(mask)

        if debug:
            print("\n", os.path.basename(self.file_path))
            print("Detected lanes:", rects)

        return rects

    def display_cropped_strip_with_rects(self):
        """
        Display the cropped strip image with detected band and grid rectangles
        overlaid for visualization.

        Raises:
            ValueError: If cropped strip is not available or detection results
            not available.
        """

        if self.cropped_strip is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )

        pad_floor = int((30 / 77) * self.cropped_strip.shape[1])
        rects = self._auto_detect_lanes(
            self.cropped_strip,
            min_lane_size=5,
            threshold_frac=self.LANE_THRESHOLD_FRAC,
            pad=pad_floor,
            debug=True,
        )
        # show the ratio between the y-position and height of the strip

        print(os.path.basename(self.file_path))
        for rect in rects:
            x, y, w, h = rect
            print(
                f"Rect y-start/H ratio: {y/self.cropped_strip.shape[0]:.3f}", end=", "
            )
            print(f"Rect y-end/H ratio: {(y+h)/self.cropped_strip.shape[0]:.3f}")

        # display the rect on the cropped strip
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(self.cropped_strip, cmap="gray")
        for rect in rects:
            x, y, w, h = rect
            rect_patch = plt.Rectangle(
                (x, y), w, h, edgecolor="red", facecolor="none", linewidth=2
            )
            ax.add_patch(rect_patch)
        plt.title("Auto-detected Lanes")
        plt.axis("on")
        plt.show()

    # Now I need to add a function that can do grid and 2 bands extraction seperately
    def extract_1grid_by_position_fallback(self, debug=False):
        if self.cropped_strip is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )

        # check if manual_detection_results is initialized
        if self.manual_detection_results is None:
            self.manual_detection_results = DetectionResults(
                band={},
                grid=[],
            )

        # GRID_TO_END_MARGIN_RATIO
        # No need dynamic pad based on image width because grid is
        # usually very clear and distinct
        # pad_floor = int((30 / 77) * self.cropped_strip.shape[1])
        pad_floor = 3

        rects = self._auto_detect_lanes(
            self.cropped_strip,
            min_lane_size=5,
            threshold_frac=self.LANE_THRESHOLD_FRAC,
            pad=pad_floor,
            debug=True,
        )

        # Usually the bottom rect is the grid area
        # But just in case, we check the position by using the GRID_TO_END_MARGIN_RATIO
        expected_grid_end_y = int(
            (1.0 - self.GRID_TO_END_MARGIN_RATIO) * self.cropped_strip.shape[0]
        )
        grid_rect = rects[-1]

        if grid_rect[1] + grid_rect[3] < expected_grid_end_y:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Detected grid position does not match expected location."
            )

        self.manual_detection_results.grid = [grid_rect]

        print(
            os.path.basename(self.file_path)
            + f" --- Detected grid at {grid_rect} and expected_grid_end_y: {expected_grid_end_y}"
        )

    def extract_2bands_by_grid_fallback(self, use_relaxed_ratio=False, debug=False):
        """
        Fallback method to extract 2 bands from the cropped strip image
        using the detected grid region as reference.

        Raises:
            ValueError: If cropped strip is not available or band counts not specified.
        """

        if self.cropped_strip is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )

        if self.band_cnts is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Band counts not specified."
            )

        if len(self.manual_detection_results.grid) == 0:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Grid region not detected. Perform grid extraction first."
            )

        grid_rect = self.manual_detection_results.grid[0]
        grid_y = grid_rect[1]
        grid_h = grid_rect[3]  # This is needed to do ratio adjustment

        if use_relaxed_ratio:
            # Use relaxed ratios, store it locally
            band0_grid_dist_ratio_range = self.BAND0_GRID_DIST_RATIO_RANGE_RELAXED
            band1_grid_dist_ratio_range = self.BAND1_GRID_DIST_RATIO_RANGE_RELAXED
            band_grid_height_ratio_range = self.BAND_GRID_HEIGHT_RATIO_RANGE_RELAXED

            lane_threshold_frac = self.LANE_THRESHOLD_FRAC_RELAXED
        else:
            band0_grid_dist_ratio_range = self.BAND0_GRID_DIST_RATIO_RANGE
            band1_grid_dist_ratio_range = self.BAND1_GRID_DIST_RATIO_RANGE
            band_grid_height_ratio_range = self.BAND_GRID_HEIGHT_RATIO_RANGE

            lane_threshold_frac = self.LANE_THRESHOLD_FRAC

        # Use the grid size to do the calculation
        adjusted_band0_grid_dist_range = (
            band0_grid_dist_ratio_range[0] * grid_h,
            band0_grid_dist_ratio_range[1] * grid_h,
        )
        adjusted_band1_grid_dist_range = (
            band1_grid_dist_ratio_range[0] * grid_h,
            band1_grid_dist_ratio_range[1] * grid_h,
        )

        adjusted_band_grid_height_range = (
            band_grid_height_ratio_range[0] * grid_h,
            band_grid_height_ratio_range[1] * grid_h,
        )

        if debug:
            print(
                os.path.basename(self.file_path)
                + f" --- Adjusted BAND0_GRID_DIST_RATIO_RANGE: {adjusted_band0_grid_dist_range}, \n"
                f"Adjusted BAND1_GRID_DIST_RATIO_RANGE: {adjusted_band1_grid_dist_range}, \n"
                f"Adjusted BAND_GRID_HEIGHT_RATIO_RANGE: {adjusted_band_grid_height_range}"
            )

        pad_floor = int((30 / 77) * self.cropped_strip.shape[1])
        rects = self._auto_detect_lanes(
            self.cropped_strip,
            min_lane_size=5,
            threshold_frac=lane_threshold_frac,
            pad=pad_floor,
            debug=True,
        )

        for rect in rects[:-1]:  # exclude the last one which is grid
            x, y, w, h = rect
            y_dist = grid_y - y
            print(y_dist, y_dist / grid_h, h / grid_h)

            if (
                y_dist >= adjusted_band0_grid_dist_range[0]
                and y_dist <= adjusted_band0_grid_dist_range[1]
                and h >= adjusted_band_grid_height_range[0]
                and h <= adjusted_band_grid_height_range[1]
            ):
                if 0 not in self.manual_detection_results.band:
                    self.manual_detection_results.band[0] = rect

                    if debug:
                        print(f"Detected 1st band at {rect}")
                else:
                    # duplicate detection, get the one that is bigger in height
                    # as it is more likely to be the correct one
                    existing_rect = self.manual_detection_results.band[0]
                    if rect[3] > existing_rect[3]:
                        self.manual_detection_results.band[0] = rect

                        if debug:
                            print(f"Updated 1st band at {rect} due to larger height")

            elif (
                y_dist >= adjusted_band1_grid_dist_range[0]
                and y_dist <= adjusted_band1_grid_dist_range[1]
                and h >= adjusted_band_grid_height_range[0]
                and h <= adjusted_band_grid_height_range[1]
            ):
                if (self.band_cnts - 1) not in self.manual_detection_results.band:
                    self.manual_detection_results.band[self.band_cnts - 1] = rect

                    if debug:
                        print(f"Detected 2nd band at {rect}")
                else:
                    # duplicate detection, get the one that is bigger in height
                    # as it is more likely to be the correct one
                    existing_rect = self.manual_detection_results.band[
                        self.band_cnts - 1
                    ]
                    if rect[3] > existing_rect[3]:
                        self.manual_detection_results.band[self.band_cnts - 1] = rect

                        if debug:
                            print(f"Updated 2nd band at {rect} due to larger height")

            else:
                print(
                    f"Skipping rectangle {rect}: Detected band position does not match expected locations."
                )

        # Check if one of the bands is detected, if not, print warning and return false
        if (
            0 not in self.manual_detection_results.band
            and (self.band_cnts - 1) not in self.manual_detection_results.band
        ):
            # print warning
            print(
                "Warning: "
                + os.path.basename(self.file_path)
                + " --- Could not detect both bands based on grid position.",
                file=sys.stderr,
            )
            return False

        # boolean to check if it overlaps with existing band
        is_overlapping = False

        # Estimate missing band position if not detected
        if (self.band_cnts - 1) not in self.manual_detection_results.band:
            # estimate 2nd band position, based on 1st band (must be here, otherwise error above)
            band0_rect = self.manual_detection_results.band[0]
            est_y = int(
                grid_y
                - (
                    adjusted_band1_grid_dist_range[0]
                    + adjusted_band1_grid_dist_range[1]
                )
                / 2
            )
            est_h = int(band0_rect[3] * 1.25)

            est_rect_o = (0, est_y, band0_rect[2], est_h)

            # Also need to check if the estimated rect start touches the end of
            # band0_rect to avoid overlapping
            if est_y < band0_rect[1] + band0_rect[3]:
                # manually adjust the est_y to avoid overlapping
                est_y = band0_rect[1] + band0_rect[3] + 1
                is_overlapping = True

            est_rect = (0, est_y, band0_rect[2], est_h)
            self.manual_detection_results.band[self.band_cnts - 1] = est_rect

            if debug:
                print(
                    os.path.basename(self.file_path)
                    + f" --- Estimated 2nd band at {est_rect_o}"
                )
                if is_overlapping:
                    print(
                        "Note: Estimated band rectangle adjusted to avoid overlapping. rect:",
                        est_rect,
                    )

        elif 0 not in self.manual_detection_results.band:
            # estimate 1st band position, based on 2nd band
            band1_rect = self.manual_detection_results.band[self.band_cnts - 1]
            est_y = int(
                grid_y
                - (
                    adjusted_band0_grid_dist_range[0]
                    + adjusted_band0_grid_dist_range[1]
                )
                / 2
            )
            est_h = int(band1_rect[3] * 1.25)
            est_rect_o = (0, est_y, band1_rect[2], est_h)

            # Also need to check if the estimated rect end touches the start of
            # band1_rect to avoid overlapping
            if est_y + est_h > band1_rect[1]:
                # manually adjust the est_y to avoid overlapping
                est_y = band1_rect[1] - est_h - 1
                is_overlapping = True

            est_rect = (0, est_y, band1_rect[2], est_h)
            self.manual_detection_results.band[0] = est_rect

            if debug:
                print(
                    os.path.basename(self.file_path)
                    + f" --- Estimated 1st band at {est_rect_o}"
                )
                if is_overlapping:
                    print(
                        "Note: Estimated band rectangle adjusted to avoid overlapping. rect:",
                        est_rect,
                    )

        return True

    def extract_2bands1grid_by_position_fallback(self, debug=False):
        """
        Fallback method to extract 2 bands and grid region from the cropped strip image
        using projection profiling, which is a technique to analyze the
        distribution of pixel intensities (lanes).

        This method doesn't use the grid to trace back the location of the bands

        Only use this method when the strip is similar to the expected size,
        which is around (660, 84)

        Raises:
            ValueError: If cropped strip is not available or band counts not specified.
        """

        if self.cropped_strip is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )

        if self.band_cnts is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Band counts not specified."
            )

        # dynamic pad based on image width
        pad_floor = int((30 / 77) * self.cropped_strip.shape[1])
        rects = self._auto_detect_lanes(
            self.cropped_strip,
            min_lane_size=5,
            threshold_frac=self.LANE_THRESHOLD_FRAC,
            pad=pad_floor,
            debug=True,
        )

        # The bottom rect is usually the grid area
        # YOLO should always give us the grid (if not the program will stop)
        # TODO: And the grid area will also be padded by pad_floor
        # So we just assume we need to get all the band and grid at once
        grid_rect = rects[-1]

        self.manual_detection_results.grid = [grid_rect]

        # Now get the bands backward
        # The 2nd band should be around 0.3-0.52 from top
        # The 1st band should be around 0.5-0.72 from top
        # and they cannot be too tall (0.1 to 0.15 height)
        band_rect_tmp = rects[-2]

        adjusted_band0_range = (self.BAND0_RATIO_RANGE[0], self.BAND0_RATIO_RANGE[1])
        adjusted_band1_range = (self.BAND1_RATIO_RANGE[0], self.BAND1_RATIO_RANGE[1])

        adjusted_band_height_range = (
            self.BAND_HEIGHT_RATIO_RANGE[0],
            self.BAND_HEIGHT_RATIO_RANGE[1],
        )

        adjusted_band0_est_y = self.BAND0_EST_Y_RATIO
        adjusted_band1_est_y = self.BAND1_EST_Y_RATIO

        if debug:
            print(
                os.path.basename(self.file_path)
                + f" --- Adjusted BAND0_RATIO_RANGE: {adjusted_band0_range}, \n"
                f"Adjusted BAND1_RATIO_RANGE: {adjusted_band1_range}, \n"
                f"Adjusted BAND_HEIGHT_RATIO_RANGE: {adjusted_band_height_range}, \n"
                f"Adjusted BAND0_EST_Y: {adjusted_band0_est_y}, \n"
                f"Adjusted BAND1_EST_Y: {adjusted_band1_est_y}"
            )

        if (
            band_rect_tmp[1] > adjusted_band1_range[0] * self.cropped_strip.shape[0]
            and band_rect_tmp[1] < adjusted_band1_range[1] * self.cropped_strip.shape[0]
            and band_rect_tmp[3]
            < adjusted_band_height_range[1] * self.cropped_strip.shape[0]
            and band_rect_tmp[3]
            > adjusted_band_height_range[0] * self.cropped_strip.shape[0]
        ):
            self.manual_detection_results.band[self.band_cnts - 1] = band_rect_tmp

            if debug:
                print(f"Detected 2nd band at {band_rect_tmp}")

        elif (
            band_rect_tmp[1] > adjusted_band0_range[0] * self.cropped_strip.shape[0]
            and band_rect_tmp[1] < adjusted_band0_range[1] * self.cropped_strip.shape[0]
            and band_rect_tmp[3]
            < adjusted_band_height_range[1] * self.cropped_strip.shape[0]
            and band_rect_tmp[3]
            > adjusted_band_height_range[0] * self.cropped_strip.shape[0]
        ):
            # this is actually the 1st band
            self.manual_detection_results.band[0] = band_rect_tmp

            if debug:
                print(f"Detected 1st band at {band_rect_tmp}")
        else:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Detected band position does not match expected locations."
            )

        # We now either get the 1st band or 2nd band, depending on what we got above
        # If we got 2nd band, we need to find 1st band, which can be rects[-3],
        # or it might not be there (0 concentration), which means the rect can be too
        # big or small. If this is the case, we need to estimate its position
        if (self.band_cnts - 1) in self.manual_detection_results.band:
            # need to find 1st band
            if len(rects) >= 3:
                band_rect_tmp = rects[-3]
                if (
                    band_rect_tmp[1]
                    > adjusted_band0_range[0] * self.cropped_strip.shape[0]
                    and band_rect_tmp[1]
                    < adjusted_band0_range[1] * self.cropped_strip.shape[0]
                    and band_rect_tmp[3]
                    < adjusted_band_height_range[1] * self.cropped_strip.shape[0]
                    and band_rect_tmp[3]
                    > adjusted_band_height_range[0] * self.cropped_strip.shape[0]
                ):
                    self.manual_detection_results.band[0] = band_rect_tmp

                    if debug:
                        print(f"Detected 1st band at {band_rect_tmp}")
                else:
                    # 0 concentration case, estimate position
                    # estimate height based on 2nd band height,
                    # and estimate y based on 0.35 ratio of the strip height
                    band1_rect = self.manual_detection_results.band[self.band_cnts - 1]
                    est_y = int(adjusted_band0_est_y * self.cropped_strip.shape[0])
                    est_h = band1_rect[3]
                    est_rect = (0, est_y, band1_rect[2], est_h)

                    self.manual_detection_results.band[0] = est_rect

                    if debug:
                        print(f"Estimated 1st band at {est_rect}")
            else:
                # Also 0 concentration case, estimate position
                # estimate height based on 2nd band height,
                # and estimate y based on 0.35 ratio of the strip height
                band1_rect = self.manual_detection_results.band[self.band_cnts - 1]
                est_y = int(adjusted_band0_est_y * self.cropped_strip.shape[0])
                est_h = band1_rect[3]
                est_rect = (0, est_y, band1_rect[2], est_h)
                self.manual_detection_results.band[0] = est_rect

                if debug:
                    print(f"Estimated 1st band at {est_rect}")

        elif 0 in self.manual_detection_results.band:
            # need to estimate 2nd band position
            band0_rect = self.manual_detection_results.band[0]
            est_y = int(adjusted_band1_est_y * self.cropped_strip.shape[0])
            est_h = band0_rect[3]
            est_rect = (0, est_y, band0_rect[2], est_h)
            self.manual_detection_results.band[self.band_cnts - 1] = est_rect

            if debug:
                print(f"Estimated 2nd band at {est_rect}")

    def display_extracted_2bands1grid_fallback(self):
        """
        Display the extracted bands and grid regions on the cropped strip image.
        Raises:
            ValueError: If cropped strip is not available or detection results
            not available.
        """

        if self.cropped_strip is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Cropped strip not available. Perform strip detection first."
            )
        if self.manual_detection_results is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- No detection results available. Perform band extraction first."
            )

        # display the rect on the cropped strip
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(self.cropped_strip, cmap="gray")
        for band_id, (x, y, w, h) in self.manual_detection_results.band.items():
            rect_patch = plt.Rectangle(
                (x, y), w, h, edgecolor="red", facecolor="none", linewidth=2
            )
            ax.add_patch(rect_patch)

        x, y, w, h = self.manual_detection_results.grid[0]
        rect_patch = plt.Rectangle(
            (x, y), w, h, edgecolor="green", facecolor="none", linewidth=2
        )
        ax.add_patch(rect_patch)

        plt.title("Detected Lanes")
        plt.axis("on")
        plt.show()

    def _single_lane_profile(self, rect, use_sum=True):
        img_gray = 1 - self.cropped_strip

        x, y, w, h = rect
        # clip to image
        H, W = img_gray.shape
        x = max(0, x)
        y = max(0, y)
        w = max(0, min(w, W - x))
        h = max(0, min(h, H - y))
        crop = img_gray[y : y + h, x : x + w]

        # Sum across rows -> intensity per column (horizontal lanes)
        if use_sum:
            profile = np.sum(crop, axis=1)  # length = w
        else:
            # ImageJ uses average, but tbh sum or mean should not matter for peak detection
            profile = np.mean(crop, axis=1)  # length = w

        positions = np.arange(y, y + h)

        return positions, profile

    def calculate_all_lanes_profiles(self, use_sum=True, debug=False):
        if self.final_detection_results is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Final detection results not available. "
                + "Conclude detection results first."
            )

        # Bands
        for band_id, rect in self.final_detection_results.band.items():
            positions, profile = self._single_lane_profile(rect, use_sum=use_sum)
            self.final_detection_results.band_pos_profile[band_id] = (
                positions,
                profile,
            )

        # Grid (there should only be one)
        self.final_detection_results.grid_pos_profile = []
        for rect in self.final_detection_results.grid:
            positions, profile = self._single_lane_profile(rect, use_sum=use_sum)
            self.final_detection_results.grid_pos_profile.append((positions, profile))

        # display for debug, sort by positions
        if debug:
            for band_id, (
                positions,
                profile,
            ) in sorted(self.final_detection_results.band_pos_profile.items()):
                plt.figure()
                plt.plot(positions, profile)
                plt.title(f"{os.path.basename(self.file_path)}: Band {band_id} Profile")
                plt.xlabel("Position along band height")
                plt.ylabel("Intensity")
                plt.show()

            for i, (positions, profile) in enumerate(
                self.final_detection_results.grid_pos_profile
            ):
                plt.figure()
                plt.plot(positions, profile)
                plt.title(f"Grid {i} Profile")
                plt.xlabel("Position along grid height")
                plt.ylabel("Intensity")
                plt.show()

    def _baseline_als(self, y, lam=1e6, p=0.001, niter=10):
        """
        Asymmetric Least Squares baseline correction.
        y: 1D array (intensity profile)
        lam: smoothness parameter (larger -> smoother baseline)
        p: asymmetry parameter (0 < p < 1). Smaller p -> baseline below peaks.
        niter: number of iterations
        returns: baseline (same length as y)
        """

        y = np.asarray(y, dtype=float)
        L = y.size
        if L < 3:
            return np.zeros_like(y)

        # Difference matrix
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
        D = D.transpose().dot(D)  # (L x L) smoothness operator
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D
            z = spsolve(Z, w * y)
            # update weights: if y > z, weight = p, else weight = (1-p)
            # Use boolean masks to avoid repeated comparisons
            diff = y - z
            w = p * (diff > 0) + (1 - p) * (diff <= 0)
            # ensure w is float
            w = w.astype(float)
        return z

    def baseline_fitting(self, baseline_params={}, debug=False):
        # Check if lane_profiles is populated
        if self.final_detection_results is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Band profiles not available. Calculate band profiles first."
            )

        if self.final_detection_results.band_pos_profile is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Band profiles not available. Calculate band profiles first."
            )

        if self.final_detection_results.grid_pos_profile is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Grid profiles not available. Calculate grid profiles first."
            )

        lam = baseline_params.get("lam", 1e6)
        p = baseline_params.get("p", 0.001)
        niter = baseline_params.get("niter", 10)

        # reset outputs so repeated calls don't accumulate
        self.final_detection_results.band_baseline = {}
        self.final_detection_results.band_corrected = {}
        self.final_detection_results.grid_baseline = []
        self.final_detection_results.grid_corrected = []

        # Bands profile baseline correction
        for band_id, (
            positions,
            profile,
        ) in self.final_detection_results.band_pos_profile.items():
            baseline = self._baseline_als(profile, lam=lam, p=p, niter=niter)
            self.final_detection_results.band_baseline[band_id] = baseline

            corrected_profile = profile - baseline
            corrected_profile = np.clip(corrected_profile, a_min=0, a_max=None)
            self.final_detection_results.band_corrected[band_id] = corrected_profile

        # Grid profile baseline correction
        for positions, profile in self.final_detection_results.grid_pos_profile:
            baseline = self._baseline_als(profile, lam=lam, p=p, niter=niter)

            self.final_detection_results.grid_baseline.append(baseline)

            corrected_profile = profile - baseline
            corrected_profile = np.clip(corrected_profile, a_min=0, a_max=None)

            self.final_detection_results.grid_corrected.append(corrected_profile)

        # display for debug, sort by positions
        if debug:
            for band_id, (
                positions,
                profile,
            ) in sorted(self.final_detection_results.band_pos_profile.items()):
                baseline = self.final_detection_results.band_baseline[band_id]
                corrected_profile = self.final_detection_results.band_corrected[band_id]

                plt.figure(figsize=(10, 6))
                plt.plot(positions, profile, label="Original Profile")
                plt.plot(positions, baseline, label="Fitted Baseline", linestyle="--")
                plt.plot(positions, corrected_profile, label="Corrected Profile")
                plt.title(
                    f"{os.path.basename(self.file_path)}: Band {band_id} Baseline Correction"
                )
                plt.xlabel("Position along band height")
                plt.ylabel("Intensity")
                plt.legend()
                plt.show()

            for i, (positions, profile) in enumerate(
                self.final_detection_results.grid_pos_profile
            ):
                baseline = self.final_detection_results.grid_baseline[i]
                corrected_profile = self.final_detection_results.grid_corrected[i]

                plt.figure(figsize=(10, 6))
                plt.plot(positions, profile, label="Original Profile")
                plt.plot(positions, baseline, label="Fitted Baseline", linestyle="--")
                plt.plot(positions, corrected_profile, label="Corrected Profile")
                plt.title(f"Grid {i} Baseline Correction")
                plt.xlabel("Position along grid height")
                plt.ylabel("Intensity")
                plt.legend()
                plt.show()

    def _three_steps_find_peaks(
        self, np_profile, min_prom_frac=0.4, est_width_factor=1.3, max_wlen=50
    ):
        # global max as the main peak location, but it cannot be too close to edges
        global_idx = int(np.argmax(np_profile))
        global_height = float(np_profile[global_idx])

        # ------------------------------------------------------------------
        # 1) Run find_peaks just to confirm the global peak
        # ------------------------------------------------------------------
        edge_margin = int(0.05 * len(np_profile))
        peaks, props = sp.signal.find_peaks(
            np_profile,
            prominence=min_prom_frac * np.max(np_profile),
        )

        # keep only interior peaks
        valid = (peaks > edge_margin) & (peaks < len(np_profile) - edge_margin)
        peaks = peaks[valid]

        # print("*******************************")
        # print( peaks, props)

        # eps = 1e-12
        # peaks, _ = sp.signal.find_peaks(
        #     np_profile,
        #     height=global_height - eps,
        #     distance=len(np_profile) + 1,  # ensures at most one peak
        # )

        if len(peaks) == 0:
            # fallback: just use the global maximum
            chosen_pk = global_idx
        else:
            # choose the highest among any candidates
            peak_candidates = np.array(peaks, dtype=int)
            candidate_vals = np_profile[peak_candidates]

            # Shouldn't need this bcos there should be only one peak
            chosen_idx_in_candidates = int(np.argmax(candidate_vals))
            chosen_pk = int(peak_candidates[chosen_idx_in_candidates])

        # ------------------------------------------------------------------
        # 2) Use peak_widths to estimate peak width, then set a local wlen
        # ------------------------------------------------------------------

        # width at half height
        widths, _, _, _ = sp.signal.peak_widths(np_profile, [chosen_pk], rel_height=0.5)
        est_width = max(float(widths[0]), 1.0)

        # search window for prominences: a few peak-widths wide, but
        # not bigger than the full profile
        wlen = int(min(len(np_profile), max(est_width_factor * est_width, max_wlen)))

        # ------------------------------------------------------------------
        # 3) Compute prominence and bases LOCALLY with wlen
        # ------------------------------------------------------------------

        proms, left_bases, right_bases = sp.signal.peak_prominences(
            np_profile, [chosen_pk], wlen=wlen
        )
        prom = float(proms[0])

        # optional: check minimum relative prominence
        if prom < min_prom_frac * global_height:
            print(
                os.path.basename(self.file_path) + f" Warning: peak prominence "
                f"{prom:.3f} < {min_prom_frac*global_height:.3f}"
            )

        lb = int(left_bases[0])
        rb = int(right_bases[0])

        return chosen_pk, lb, rb, est_width, prom, wlen

    def find_peaks_in_profiles(
        self, min_prom_frac=0.4, est_width_factor=1.3, max_wlen=50, debug=False
    ):
        if (
            self.final_detection_results is None
            or self.final_detection_results.band_corrected is None
            or self.final_detection_results.band_pos_profile is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Corrected band profiles not available. Perform baseline fitting first."
            )

        if (
            self.final_detection_results.grid_corrected is None
            or self.final_detection_results.grid_pos_profile is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Corrected grid profiles not available. Perform baseline fitting first."
            )

        # Bands profile peak finding
        for (
            band_id,
            corrected_profile,
        ) in self.final_detection_results.band_corrected.items():
            np_corrected_profile = np.asarray(corrected_profile, dtype=float)

            chosen_pk, lb, rb, est_width, prom, wlen = self._three_steps_find_peaks(
                np_corrected_profile,
                min_prom_frac=min_prom_frac,
                est_width_factor=est_width_factor,
                max_wlen=max_wlen,
            )

            if self.final_detection_results.band_corrected_lbs is None:
                self.final_detection_results.band_corrected_lbs = {}
            if self.final_detection_results.band_corrected_rbs is None:
                self.final_detection_results.band_corrected_rbs = {}

            self.final_detection_results.band_corrected_lbs[band_id] = lb
            self.final_detection_results.band_corrected_rbs[band_id] = rb

            # area debug / sanity print
            # xs = positions[lb : rb + 1]
            # ys = profile[lb : rb + 1]
            positions = self.final_detection_results.band_pos_profile[band_id][0]

            print(
                f"[band {band_id}] peak at x={positions[chosen_pk]}, "
                f"Estimated peak width = {est_width}"
                f"lb={positions[lb]}, rb={positions[rb]}, "
                f"prom={prom:.3f}, width={rb - lb} vs wlen={wlen} "
            )

        self.final_detection_results.grid_corrected_lbs = []
        self.final_detection_results.grid_corrected_rbs = []

        # Grid profile peak finding
        for band_id, corrected_profile in enumerate(
            self.final_detection_results.grid_corrected
        ):
            np_corrected_profile = np.asarray(corrected_profile, dtype=float)

            chosen_pk, lb, rb, est_width, prom, wlen = self._three_steps_find_peaks(
                np_corrected_profile,
                min_prom_frac=min_prom_frac,
                est_width_factor=est_width_factor,
                max_wlen=max_wlen,
            )

            self.final_detection_results.grid_corrected_lbs.append(lb)
            self.final_detection_results.grid_corrected_rbs.append(rb)

            positions = self.final_detection_results.grid_pos_profile[band_id][0]

            print(
                f"[grid] peak at x={positions[chosen_pk]}, "
                f"Estimated peak width = {est_width}"
                f"lb={positions[lb]}, rb={positions[rb]}, "
                f"prom={prom:.3f}, width={rb - lb} vs wlen={wlen} "
            )

        if debug:
            for band_id, (
                positions,
                _,
            ) in sorted(self.final_detection_results.band_pos_profile.items()):
                corrected_profile = self.final_detection_results.band_corrected[band_id]
                lb = self.final_detection_results.band_corrected_lbs[band_id]
                rb = self.final_detection_results.band_corrected_rbs[band_id]

                plt.figure(figsize=(10, 6))
                plt.plot(
                    positions,
                    corrected_profile,
                    color="green",
                    label="Corrected Profile",
                )

                ax = plt.gca()
                ax.axvline(
                    x=positions[lb], color="orange", linestyle="--", label="Left Base"
                )
                ax.axvline(
                    x=positions[rb], color="purple", linestyle="--", label="Right Base"
                )
                plt.title(
                    f"{os.path.basename(self.file_path)}: Band {band_id} Peak Detection"
                )
                plt.xlabel("Position along band height")
                plt.ylabel("Intensity")
                plt.legend()
                plt.show()

            for grid_id, (
                positions,
                _,
            ) in enumerate(self.final_detection_results.grid_pos_profile):
                corrected_profile = self.final_detection_results.grid_corrected[grid_id]
                lb = self.final_detection_results.grid_corrected_lbs[grid_id]
                rb = self.final_detection_results.grid_corrected_rbs[grid_id]

                plt.figure(figsize=(10, 6))
                plt.plot(
                    positions,
                    corrected_profile,
                    color="green",
                    label="Corrected Profile",
                )

                ax = plt.gca()
                ax.axvline(
                    x=positions[lb], color="orange", linestyle="--", label="Left Base"
                )
                ax.axvline(
                    x=positions[rb], color="purple", linestyle="--", label="Right Base"
                )
                plt.title("Grid Peak Detection")
                plt.xlabel("Position along grid height")
                plt.ylabel("Intensity")
                plt.legend()
                plt.show()

    def calculate_peak_areas(self, debug=False):
        # Calculate the area for the band peaks
        if (
            self.final_detection_results is None
            or self.final_detection_results.band_corrected is None
            or self.final_detection_results.band_corrected_lbs is None
            or self.final_detection_results.band_corrected_rbs is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Corrected band profiles or peak bases not available. "
                + "Perform peak finding first."
            )

        self.final_detection_results.band_corrected_peak_areas = {}

        for (
            band_id,
            corrected_profile,
        ) in self.final_detection_results.band_corrected.items():
            lb = self.final_detection_results.band_corrected_lbs[band_id]
            rb = self.final_detection_results.band_corrected_rbs[band_id]

            # integrate area under the curve between lb and rb
            area = np.trapz(corrected_profile[lb : rb + 1])

            self.final_detection_results.band_corrected_peak_areas[band_id] = area

            print(
                os.path.basename(self.file_path)
                + f" --- [band {band_id}] Peak area: {area:.3f}"
            )

        # Calculate the area for the grid peaks
        if (
            self.final_detection_results.grid_corrected is None
            or self.final_detection_results.grid_corrected_lbs is None
            or self.final_detection_results.grid_corrected_rbs is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Corrected grid profiles or peak bases not available. "
                + "Perform peak finding first."
            )

        self.final_detection_results.grid_corrected_peak_areas = []

        for grid_id, corrected_profile in enumerate(
            self.final_detection_results.grid_corrected
        ):
            lb = self.final_detection_results.grid_corrected_lbs[grid_id]
            rb = self.final_detection_results.grid_corrected_rbs[grid_id]

            # integrate area under the curve between lb and rb
            area = np.trapz(corrected_profile[lb : rb + 1])

            self.final_detection_results.grid_corrected_peak_areas.append(area)

            if debug:
                print(
                    os.path.basename(self.file_path)
                    + f" [grid {grid_id}] Peak area: {area:.3f}\n"
                )

    def get_band_peak_area(self, band_id=0):
        if (
            self.final_detection_results is None
            or self.final_detection_results.band_corrected_peak_areas is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Band peak areas not available. Perform peak area calculation first."
            )

        if band_id not in self.final_detection_results.band_corrected_peak_areas:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + f" --- Band ID {band_id} peak area not available."
            )

        return self.final_detection_results.band_corrected_peak_areas[band_id]

    def get_grid_peak_area(self, grid_id=0):
        if (
            self.final_detection_results is None
            or self.final_detection_results.grid_corrected_peak_areas is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Grid peak areas not available. Perform peak area calculation first."
            )

        # There should only be one grid, but just in case
        if grid_id >= len(self.final_detection_results.grid_corrected_peak_areas):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + f" --- Grid ID {grid_id} peak area not available."
            )

        return self.final_detection_results.grid_corrected_peak_areas[grid_id]

    def get_filename(self):
        return os.path.basename(self.file_path)

    def get_band_peak_areas_mapping(self):
        if (
            self.final_detection_results is None
            or self.final_detection_results.band_corrected_peak_areas is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Band peak areas not available. Perform peak area calculation first."
            )

        return self.final_detection_results.band_corrected_peak_areas

    def get_grid_peak_areas_list(self):
        if (
            self.final_detection_results is None
            or self.final_detection_results.grid_corrected_peak_areas is None
        ):
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Grid peak areas not available. Perform peak area calculation first."
            )

        return self.final_detection_results.grid_corrected_peak_areas

    def export_band_pos_profile_csv(
        self, csv_path: str, include_grid: bool = False
    ) -> None:
        """
        Export band_pos_profile (and optionally grid_pos_profile) to a CSV file.

        CSV layout:
        For each band:
            # Band <id>
            Position,Intensity
            pos0,int0
            pos1,int1
            ...
        (blank line)
        For grid (if include_grid=True), each grid block follows:
            # Grid <idx>
            Position,Intensity
            pos0,int0
            ...

        Args:
            csv_path: Output CSV file path (string).
            include_grid: Whether to append grid_pos_profile after band blocks.
        Raises:
            ValueError: If final_detection_results or band_pos_profile is missing.
            IOError: If file cannot be written.
        """

        # Validation
        if self.final_detection_results is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Final detection results not available. "
                + "Conclude detection results first."
            )

        if self.final_detection_results.band_pos_profile is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- band_pos_profile not available. "
                + "Calculate band profiles first."
            )

        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)

                # Write header comment with filename
                writer.writerow(
                    [
                        f"# Exported band_pos_profile for: {os.path.basename(self.file_path)}"
                    ]
                )
                writer.writerow([])

                # Bands (sorted by band id)
                for band_id in sorted(
                    self.final_detection_results.band_pos_profile.keys()
                ):
                    positions, profile = self.final_detection_results.band_pos_profile[
                        band_id
                    ]
                    writer.writerow([f"Band {band_id}"])
                    writer.writerow(["Position", "Intensity"])
                    # positions and profile are 1-D arrays or lists of same length
                    for pos, val in zip(positions, profile):
                        writer.writerow([int(pos), float(val)])
                    writer.writerow([])  # blank line between bands

                if include_grid:
                    if self.final_detection_results.grid_pos_profile is None:
                        # No grid profiles available, warn and skip
                        writer.writerow(["# No grid_pos_profile available"])
                    else:
                        for grid_idx, (positions, profile) in enumerate(
                            self.final_detection_results.grid_pos_profile
                        ):
                            writer.writerow([f"Grid {grid_idx}"])
                            writer.writerow(["Position", "Intensity"])
                            for pos, val in zip(positions, profile):
                                writer.writerow([int(pos), float(val)])
                            writer.writerow([])

        except Exception as e:
            raise IOError(f"Failed to write CSV to {csv_path}: {e}")

    def export_band_pos_corrected_profile_csv(
        self, csv_path: str, include_grid: bool = False
    ) -> None:
        # Validation
        if self.final_detection_results is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- Final detection results not available. "
                + "Conclude detection results first."
            )

        if self.final_detection_results.band_pos_profile is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- band_pos_profile not available. "
                + "Calculate band profiles first."
            )

        if self.final_detection_results.band_corrected is None:
            raise ValueError(
                "Error: "
                + os.path.basename(self.file_path)
                + " --- band_corrected not available. "
                + "Perform baseline fitting first."
            )

        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)

                # Write header comment with filename
                writer.writerow(
                    [
                        f"# Exported band_corrected profiles for: {os.path.basename(self.file_path)}"
                    ]
                )
                writer.writerow([])

                # Bands (sorted by band id)
                for band_id in sorted(
                    self.final_detection_results.band_pos_profile.keys()
                ):
                    positions, _ = self.final_detection_results.band_pos_profile[
                        band_id
                    ]
                    corrected_profile = self.final_detection_results.band_corrected[
                        band_id
                    ]

                    writer.writerow([f"Band {band_id} Corrected Profile"])
                    writer.writerow(["Position", "Corrected Intensity"])

                    for pos, val in zip(positions, corrected_profile):
                        writer.writerow([int(pos), float(val)])
                    writer.writerow([])  # blank line between bands

                if include_grid:
                    if (
                        self.final_detection_results.grid_pos_profile is None
                        or self.final_detection_results.grid_corrected is None
                    ):
                        # No grid profiles available, warn and skip
                        writer.writerow(
                            ["# No grid_pos_profile or grid_corrected available"]
                        )
                    else:
                        for grid_idx, (positions, _) in enumerate(
                            self.final_detection_results.grid_pos_profile
                        ):
                            corrected_profile = (
                                self.final_detection_results.grid_corrected[grid_idx]
                            )

                            writer.writerow([f"Grid {grid_idx} Corrected Profile"])
                            writer.writerow(["Position", "Corrected Intensity"])

                            for pos, val in zip(positions, corrected_profile):
                                writer.writerow([int(pos), float(val)])
                            writer.writerow([])

        except Exception as e:
            raise IOError(f"Failed to write CSV to {csv_path}: {e}")

    def plot_intensity_baseline_polished(self):
        # from matplotlib.ticker import MaxNLocator
        from matplotlib.ticker import MultipleLocator

        enlarge_size = 2.5

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 16 * enlarge_size,
                "axes.labelsize": 22 * enlarge_size,
                "axes.titlesize": 22 * enlarge_size,
                "axes.linewidth": 2 * enlarge_size,
                "xtick.labelsize": 22 * enlarge_size,
                "ytick.labelsize": 22 * enlarge_size,
                "legend.fontsize": 22 * enlarge_size,
                "lines.linewidth": 3 * enlarge_size,
                "figure.facecolor": "none",
                "axes.facecolor": "none",
            }
        )

        band_colors = {
            0: "#CCA000",
            1: "#C33B34",
        }

        grid_colors = {
            0: "#548235",
        }

        baseline_color = "#7F7F7F"

        # -----------------------------
        # Plot Bands
        # -----------------------------
        for band_id, (positions, profile) in sorted(
            self.final_detection_results.band_pos_profile.items()
        ):
            baseline = self.final_detection_results.band_baseline[band_id]
            color = band_colors.get(band_id, "#333333")

            fig, ax = plt.subplots(figsize=(17, 7))

            ax.plot(positions, profile, label="Intensity profile", color=color)

            ax.plot(
                positions,
                baseline,
                linestyle="--",
                color=baseline_color,
                label="Fitted baseline",
            )

            ax.set_xlabel("Position along band height", labelpad=15)
            ax.set_ylabel("Intensity (a.u.)", labelpad=15)
            # ax.set_title(f"{basename} | Band {band_id}")

            # Force ticks
            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(2))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Legend inside plot, top-right with slight offset
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.4, 1),
                frameon=True,
                facecolor="none",
                edgecolor="none",
            )

            plt.show()

        # -----------------------------
        # Plot Grids
        # -----------------------------
        for i, (positions, profile) in enumerate(
            self.final_detection_results.grid_pos_profile
        ):
            baseline = self.final_detection_results.grid_baseline[i]
            color = grid_colors.get(i, "#333333")

            fig, ax = plt.subplots(figsize=(17, 7))

            ax.plot(positions, profile, label="Intensity profile", color=color)

            ax.plot(
                positions,
                baseline,
                linestyle="--",
                color=baseline_color,
                label="Fitted baseline",
            )

            ax.set_xlabel("Position along grid height", labelpad=15)
            ax.set_ylabel("Intensity (a.u.)", labelpad=15)
            # ax.set_title(f"{basename} | Grid {i}")

            # Force ticks
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(20))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.4, 1),
                frameon=True,
                facecolor="none",
                edgecolor="none",
            )

            plt.show()

    def plot_area_lrb(self):
        from matplotlib.ticker import MultipleLocator

        # Get the peak areas for bands and grids
        normalized_peak_areas = self.get_band_peak_areas_mapping()
        grid_areas = self.get_grid_peak_areas_list()

        enlarge_size = 2.5

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 16 * enlarge_size,
                "axes.labelsize": 22 * enlarge_size,
                "axes.titlesize": 22 * enlarge_size,
                "axes.linewidth": 2 * enlarge_size,
                "xtick.labelsize": 22 * enlarge_size,
                "ytick.labelsize": 22 * enlarge_size,
                "legend.fontsize": 22 * enlarge_size,
                "lines.linewidth": 3 * enlarge_size,
                "figure.facecolor": "none",
                "axes.facecolor": "none",
            }
        )

        band_colors = {
            0: "#CCA000",
            1: "#C33B34",
        }

        grid_colors = {
            0: "#548235",
        }

        base_color = "#7F7F7F"
        fill_alpha = 0.2

        # -----------------------------
        # Bands
        # -----------------------------
        for band_id, (positions, _) in sorted(
            self.final_detection_results.band_pos_profile.items()
        ):
            corrected_profile = self.final_detection_results.band_corrected[band_id]
            lb = int(self.final_detection_results.band_corrected_lbs[band_id])
            rb = int(self.final_detection_results.band_corrected_rbs[band_id])

            color = band_colors.get(band_id, "#333333")

            pos_arr = np.asarray(positions)
            prof_arr = np.asarray(corrected_profile)

            left_pos = pos_arr[lb]
            right_pos = pos_arr[rb]

            mask = (pos_arr >= left_pos) & (pos_arr <= right_pos)

            fig, ax = plt.subplots(figsize=(19, 7))

            ax.plot(
                pos_arr,
                prof_arr,
                label="Corrected profile",
                color=color,
            )

            ax.axvline(
                x=left_pos,
                linestyle="--",
                color=base_color,
                label="Left & right base",
            )

            # Fill area down to x-axis
            ax.fill_between(
                pos_arr[mask],
                prof_arr[mask],
                0,
                facecolor=color,
                alpha=fill_alpha,
                label=rf"Peak area = {normalized_peak_areas[band_id]:.2f}",
            )

            ax.axvline(
                x=right_pos,
                linestyle="--",
                color=base_color,
                label="_nolegend_",
            )

            ax.set_xlabel("Position along band height", labelpad=15)
            ax.set_ylabel("Intensity (a.u.)", labelpad=15)

            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_major_locator(MultipleLocator(2))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.65, 1),
                frameon=True,
                facecolor="none",
                edgecolor="none",
            )

            plt.show()

        # -----------------------------
        # Grids
        # -----------------------------
        for grid_id, (positions, _) in enumerate(
            self.final_detection_results.grid_pos_profile
        ):
            corrected_profile = self.final_detection_results.grid_corrected[grid_id]
            lb = int(self.final_detection_results.grid_corrected_lbs[grid_id])
            rb = int(self.final_detection_results.grid_corrected_rbs[grid_id])

            color = grid_colors.get(grid_id, "#333333")

            pos_arr = np.asarray(positions)
            prof_arr = np.asarray(corrected_profile)

            left_pos = pos_arr[lb]
            right_pos = pos_arr[rb]

            mask = (pos_arr >= left_pos) & (pos_arr <= right_pos)

            fig, ax = plt.subplots(figsize=(19, 7))

            ax.plot(
                pos_arr,
                prof_arr,
                label="Corrected profile",
                color=color,
            )

            ax.axvline(
                x=left_pos,
                linestyle="--",
                color=base_color,
                label="Left & right base",
            )

            ax.fill_between(
                pos_arr[mask],
                prof_arr[mask],
                0,
                facecolor=color,
                alpha=fill_alpha,
                label=rf"Peak area = {grid_areas[0]:.2f}",
            )

            ax.axvline(
                x=right_pos,
                linestyle="--",
                color=base_color,
                label="_nolegend_",
            )

            ax.set_xlabel("Position along grid height", labelpad=15)
            ax.set_ylabel("Intensity (a.u.)", labelpad=15)

            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(20))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.6, 1),
                frameon=True,
                facecolor="none",
                edgecolor="none",
            )

            plt.show()

        for band_id in normalized_peak_areas:
            normalized_peak_areas[band_id] = (
                normalized_peak_areas[band_id] / grid_areas[0]
            )

        return normalized_peak_areas
