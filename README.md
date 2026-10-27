# 🧪 📷 MOSAIC: Machine-Learning-Assisted Image Analysis

Park, C.; Dissanayake-Perera, S.; Ng, H.-C.; Sadler, C. J.; Gallego Rubio, C.; Nam, K.; Prasad, S. K.; Stevens, M. M. Nanozyme-Catalyzed Lateral Flow Assay for the Simultaneous Detection of RNA and Protein Biomarkers. _ACS Nano_, in press (2026).

**If you use this repository in your research, please cite the associated MOSAIC publication.**

## 1. Overview

This repository contains the Python code developed for the machine-learning-assisted quantitative interpretation of the MOSAIC (Multiplexed Orthogonal Sensing Assay Integrated with CRISPRZyme) lateral flow assay.

The computational workflow integrates:

- YOLOv8-based object detection
- Automated test strip and band localization
- Geometric estimation of missing test-band regions
- One-dimensional intensity profile extraction
- Asymmetric least-squares (ALS) baseline correction
- Peak detection and numerical integration
- Reference-grid normalization
- Four-parameter logistic (4PL) calibration


The analysis was developed using **TensorFlow 2.15** in a Docker-based Jupyter Notebook environment. Docker is recommended to provide a consistent execution environment.

## 2. Running the Docker Container

### Prerequisites

Install [Docker](https://docs.docker.com/get-docker/) on your computer.

For GPU acceleration, an NVIDIA GPU with compatible drivers and the NVIDIA Container Toolkit is required. GPU acceleration is optional for inference.

### Step 1: Clone the repository

```bash
git clone https://github.com/The-Stevens-Group/MOSAIC
cd MOSAIC
```

### Step 2: Start the TensorFlow Docker container

**GPU-enabled:**

```bash
docker run --gpus all -it --rm \
  -v "$(pwd):/tf/notebooks" \
  -p 8888:8888 \
  tensorflow/tensorflow:2.15.0-gpu-jupyter
```

**CPU-only:**

```bash
docker run -it --rm \
  -v "$(pwd):/tf/notebooks" \
  -p 8888:8888 \
  tensorflow/tensorflow:2.15.0-jupyter
```

The `$(pwd)` expression automatically resolves to the current repository directory.

The repository is mounted at `/tf/notebooks` inside the Docker container.


### Step 3: Access Jupyter Notebook

Once the container starts, open the following address in your browser:

**http://localhost:8888**

Use the authentication token displayed in the terminal to access Jupyter Notebook.


### Step 4: Install additional dependencies

Open `/src/install_tf2.15.ipynb` and run the notebook to install all required dependencies. You can safely ignore any error messages displayed in the output.

## 3. Data Format

_Note: This section can be skipped if you are only interested in running or exploring the code and computational pipeline. The following data format details are primarily intended for users who wish to use their own datasets._

### 3.1 Experimental Image Data

The experimental strip images used for the quantitative analysis in Figure 6 are located in:

```text
data/Figure6/
```

The directory contains the lateral flow assay images from Figure 6 and the corresponding concentration metadata.

The expected directory structure is:

```text
data/
└── Figure6/
    ├── img_conc.csv
    ├── image_001.jpg
    ├── image_002.jpg
    ├── image_003.jpg
    └── ...
```

Images should be provided in JPEG format (`.jpg` or `.jpeg`).

During inference, images are converted to three-channel RGB and resized to 640 × 640 pixels with aspect-ratio-preserving padding.

### 3.2 Concentration Metadata

The `img_conc.csv` file maps each image to its corresponding RNA and protein concentrations.

The required CSV columns are:

| Column | Description |
|--------|-------------|
| `filename` | Filename of the corresponding strip image |
| `concentration0` | ORF1ab RNA concentration (band 0) |
| `concentration1` | S protein concentration (band 1) |

Example CSV format (illustrative values only):

```csv
filename,concentration0,concentration1
image_001.jpg,0,0
image_002.jpg,0,0
image_003.jpg,64,6.4
image_004.jpg,64,6.4
```

For the Figure 6 dataset:

- ORF1ab RNA concentrations are expressed in fM.
- S protein concentrations are expressed in pM.

Each filename must correspond to an existing image in the same directory.

Images with identical RNA and protein concentration combinations are grouped as replicate measurements during analysis.

### 3.3 YOLO Training Data

The YOLO training dataset is stored under:

```text
data/lfa_single_ds/
```

The object detection annotations use the COCO JSON format, containing image metadata, object categories, and bounding-box coordinates.

The annotation file expected by the training notebook is:

```text
data/lfa_single_ds/lfa_single_coco.json
```

Bounding boxes are stored in COCO format:

```text
[x, y, width, height]
```

The object detection model recognizes five classes:

| COCO Category ID | Object Class |
|------------------|--------------|
| 1 | Protein line |
| 2 | RNA line |
| 3 | Strip |
| 4 | Control grid line |
| 5 | Control line |

The training code converts COCO category IDs to zero-based class indices and bounding boxes to `[x_min, y_min, x_max, y_max]` coordinates.

The training notebook expects image files in subdirectories of `data/lfa_single_ds/`, with filenames matching the COCO annotations.

## 4. Reproducing the Quantitative Analysis (Figure 6)

The main analysis notebook is:

**`mosaic_main_ml.ipynb`**

This notebook performs the complete machine-learning-assisted quantitative analysis.


### Step 1: Configure the input data

The notebook uses:

```python
img_conc_csv = "../data/Figure6/img_conc.csv"
```

Ensure that the CSV file and corresponding images are available at this location.


### Step 2: Load the trained YOLO model

The trained model checkpoint used for the analysis is:

```text
saved_mdl/detect_yolo_xs/yolo_kfold_4.h5
```

Retraining YOLO is not required to reproduce the quantitative analysis.


### Step 3: Run the analysis

Execute the notebook cells sequentially.

The workflow performs the following steps:

1. Reads the concentration metadata and groups replicate images.
2. Performs YOLO-assisted strip, test-line, and reference-grid detection.
3. Estimates missing test-band regions using geometric relationships where applicable.
4. Extracts one-dimensional intensity profiles.
5. Performs ALS baseline correction and peak-area integration.
6. Normalizes test-line peak areas against the reference grid.
7. Fits separate 4PL calibration curves for ORF1ab RNA and S protein.


### Step 4: Obtain the results

The notebook generates calibration plots and exports measured and fitted data.

Exported calibration files are saved to:

```text
saved_mdl/4pl/
```

These include observed calibration measurements and fitted 4PL curves suitable for further analysis or visualization in GraphPad Prism.

**Note:** The notebook uses relative paths. The supplied configuration expects the notebook to run from a directory one level below the repository root, with `data/` and `saved_mdl/` located at the root. Adjust these paths if the repository structure differs.


## 5. YOLOv8 Model Training

The notebook:

**`detect_yolo_kfold.ipynb`**

contains the implementation of YOLOv8 object detection model training and five-fold cross-validation.

It includes:

- Dataset preprocessing
- Data augmentation
- YOLOv8 model training
- Five-fold cross-validation
- Model evaluation
- Model checkpoint saving

This notebook is provided for reproducibility of the object detection model development.

**Running the training notebook is NOT required to reproduce the Figure 6 quantitative analysis**.

The training notebook uses its own relative dataset and output paths, which may require adjustment depending on its location within the repository.


## 6. Data Availability

The raw research data supporting the MOSAIC study are available through Zenodo:

https://doi.org/10.5281/zenodo.22900029
