from .BasicStripWithGrid import BasicStripWithGrid

from typing import Optional, Any
import numpy as np

import os
import sys


# Normally 3 strips (BasicStripWithGrid) per sample
class StripSample:
    """
    Docstring for StripSample
    Represents a sample containing multiple strips with associated a set of
    concentration data (2 concentrations with 2 bands, keys 0 and 1).

    Attributes:
        concentration (dict): A dictionary mapping band indices to their
                                corresponding concentrations.
        basic_strip_w_grid_ls (list): A list of BasicStripWithGrid objects
                                        associated with this sample.
        fallback_detection_method (str):
                            Method used for conventional fallback band detection.
                        
                            "grid_relative":
                                Locate bands using the detected reference grid
                                position and height.
                        
                            "strip_height_relative":
                                Locate bands using expected positions relative
                                to the overall strip image height.
                        
                            Default is "grid_relative".

    Methods:
        create_add_basic_strip(file_path: str, concentration: dict = None):
            Creates and adds a BasicStripWithGrid object to the sample.

        append_basic_strip(bswg: BasicStripWithGrid.BasicStripWithGrid):
            Appends an existing BasicStripWithGrid object to the sample.

        run_find_peak_area_all_strips(debug: bool = False):
            Runs peak area detection for all strips in the sample.
    """

    def __init__(
        self,
        # sample_id: int,
        concentration: dict,  # e.g., {0: 50, 1: 100}
        fallback_detection_method: str = "grid_relative",
        basic_strip_w_grid_ls: Optional[list[Any]] = None,
        ml_mdl: Any = None,
        ml_threshold: Optional[float] = 0.4,
        ml_grid_pad: Optional[int] = 5,
        ml_band_pad: Optional[int] = 15,
        ml_band_estimation_pad: Optional[int] = 15,
    ):
        # self.sample_id = sample_id
        self.concentration = concentration
        self.basic_strip_w_grid_ls = (
            basic_strip_w_grid_ls if basic_strip_w_grid_ls is not None else []
        )
        
        if fallback_detection_method not in (
            "grid_relative",
            "strip_height_relative",
        ):
            raise ValueError(
                f"Invalid fallback detection method: {fallback_detection_method}"
            )
        
        self.fallback_detection_method = fallback_detection_method

        self.mean = 0.0
        self.stddev = 0.0

        self.ml_mdl = ml_mdl

        self.ml_threshold = ml_threshold
        self.ml_grid_pad = ml_grid_pad
        self.ml_band_pad = ml_band_pad
        self.ml_band_estimation_pad = ml_band_estimation_pad

    def create_add_basic_strip(
        self, file_path: str, concentration: Optional[dict[Any, Any]] = None
    ):
        # Check if the concentration provided matches the existing one
        if concentration is not None and concentration != self.concentration:
            raise ValueError(
                "Provided concentration does not match the existing concentration."
            )

        # Get the number of bands from the concentration dictionary
        num_bands = len(self.concentration)

        bswg = BasicStripWithGrid(
            file_path,
            concentration=self.concentration,
            band_cnts=num_bands,
            ml_mdl=self.ml_mdl,
        )
        self.basic_strip_w_grid_ls.append(bswg)

    def append_basic_strip(self, bswg: BasicStripWithGrid):
        self.basic_strip_w_grid_ls.append(bswg)

    def run_find_peak_area_all_strips_ml(self, debug: bool = False):
        for bswg in self.basic_strip_w_grid_ls:
            bswg.load_img()

            # Crop the strip using ML model, if not sucess, it will give error
            # Don't use fallback method here
            bswg.strip_detect_crop_ml(threshold=self.ml_threshold, debug=debug)

            # Extract grid using ml
            bswg.extract_1grid_ml(ml_pad=self.ml_grid_pad, debug=debug)

            # Extract bands using ml
            extract_band_success = bswg.extract_2bands_ml(
                ml_pad=self.ml_band_pad,
                estimation_pad=self.ml_band_estimation_pad,
                debug=debug,
            )

            if not extract_band_success:
                # print warning
                print(
                    "Warning: "
                    + os.path.basename(bswg.file_path)
                    + " --- ML band extraction failed, trying fallback method...",
                    file=sys.stderr,
                )

                self._find_bands_fallback(bswg, debug=debug)

            bswg.conclude_detection_results()

            bswg.calculate_all_lanes_profiles(use_sum=True, debug=debug)
            bswg.baseline_fitting(debug=debug)
            bswg.find_peaks_in_profiles(debug=debug)

            bswg.calculate_peak_areas(debug=debug)

            print(
                "\nCompleted peak area calculation for all strips "
                + "in the sample with ML + fallback method."
            )

    def run_find_peak_area_all_strips_fallback(self, debug: bool = False):
        for bswg in self.basic_strip_w_grid_ls:
            bswg.load_img()

            # The fallback method for cropping strip
            bswg.strip_detect_crop_fallback()
            if debug:
                bswg.display_cropped_strip_with_rects()

            self._find_bands_fallback(bswg, debug=debug)

            bswg.conclude_detection_results()

            bswg.calculate_all_lanes_profiles(use_sum=True, debug=debug)
            bswg.baseline_fitting(debug=debug)
            bswg.find_peaks_in_profiles(debug=debug)

            bswg.calculate_peak_areas(debug=debug)

        print(
            "\nCompleted peak area calculation for all strips "
            + "in the sample with fallback method."
        )

    def _find_bands_fallback(self, bswg: BasicStripWithGrid, debug: bool = False):
        # Extract bands based on grid position/height or cropped strip height ratio
        # Grid position/thickness method is more reliable because it uses the
        # height of the grid which can stretch or shrink depending on the camera
        # angle, but the grid height remains constant in reality.
        #
        # Using the height ratio of the cropped strip image can be less
        # reliable because the strip
        # height can vary due to the manual cropping. So only use it when
        # cropped strip image is with around shape: (660, 84)

        if self.fallback_detection_method == "grid_relative":
            bswg.extract_1grid_by_position_fallback(debug=debug)

            did_extract_band = bswg.extract_2bands_by_grid_fallback(debug=debug)

            if did_extract_band is False:
                bswg.extract_2bands_by_grid_fallback(use_relaxed_ratio=True, debug=True)
                
        elif self.fallback_detection_method == "strip_height_relative":
            bswg.extract_2bands1grid_by_position_fallback(debug=debug)

        if debug:
            bswg.display_extracted_2bands1grid_fallback()

    def _get_normalized_peak_areas(self, bswg: BasicStripWithGrid):
        """
        Get the normalized peak areas for a given band across one strip in the sample.

        Args:
            bswg (BasicStripWithGrid): The BasicStripWithGrid object to get peak areas from.

        Returns:
            list: A list of normalized peak areas for the specified band across all strips.
        """
        normalized_peak_areas = {}
        normalized_peak_areas = bswg.get_band_peak_areas_mapping()

        grid_areas = bswg.get_grid_peak_areas_list()

        if len(grid_areas) != 1:
            raise ValueError("Expected exactly one grid area for normalization.")
        grid_area = grid_areas[0]

        for band_id in normalized_peak_areas:
            normalized_peak_areas[band_id] = normalized_peak_areas[band_id] / grid_area

        return normalized_peak_areas

    def get_normalized_peak_areas_across_bands(self, multiply_by_100: bool = False):
        peak_areas_across_band: dict[int, list[float]] = {}

        for bswg in self.basic_strip_w_grid_ls:
            normalized_peak_areas = self._get_normalized_peak_areas(bswg)

            for band_id, area in normalized_peak_areas.items():
                if band_id not in peak_areas_across_band:
                    peak_areas_across_band[band_id] = []
                if multiply_by_100:
                    area *= 100
                area = float(area)
                peak_areas_across_band[band_id].append(area)

        return peak_areas_across_band

    def get_mean_sd_peak_areas(self, multiply_by_100: bool = False):
        """
        Calculate the mean and standard deviation of normalized peak areas
        for each band across all strips in the sample.

        Returns:
            dict: A dictionary with band indices as keys and tuples of
                  (mean, standard deviation) as values.
        """
        # band_peak_areas = {}
        peak_areas_across_band: dict[int, list[float]] = {}

        for bswg in self.basic_strip_w_grid_ls:
            normalized_peak_areas = self._get_normalized_peak_areas(bswg)

            for band_id, area in normalized_peak_areas.items():
                if band_id not in peak_areas_across_band:
                    peak_areas_across_band[band_id] = []

                if multiply_by_100:
                    area *= 100
                peak_areas_across_band[band_id].append(area)

        band_stats = {}
        for band_id, areas in peak_areas_across_band.items():
            mean_area = np.mean(areas)
            sd_area = np.std(areas)
            band_stats[band_id] = (mean_area, sd_area)

        print(
            f"\nMean and Standard Deviation of Normalized Peak Areas per Band for concentrations {self.concentration}:"
        )
        for band_id, stats in band_stats.items():
            print(f"Band {band_id}: Mean = {stats[0]:.4f}, SD = {stats[1]:.4f}")

        return band_stats

    def get_num_strips(self) -> int:
        """
        Get the number of strips in the sample.

        Returns:
            int: The number of BasicStripWithGrid objects in the sample.
        """
        return len(self.basic_strip_w_grid_ls)

    def get_concentration(self) -> dict:
        """
        Get the concentration mapping for the sample.

        Returns:
            dict: The concentration dictionary mapping band indices to concentrations.
        """
        return self.concentration
