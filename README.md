# MOSAIC: Machine-Learning-Assisted Image Analysis

Park, C.; Dissanayake-Perera, S.; Ng, H.-C.; Sadler, C. J.; Gallego Rubio, C.; Nam, K.; Prasad, S. K.; Stevens, M. M. Nanozyme-Catalyzed Lateral Flow Assay for the Simultaneous Detection of RNA and Protein Biomarkers. _ACS Nano_, in press (2026).

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

Open `/src/install_tf2.15.ipynb` to install all the needed dependencies. Ignore the error messages in the output.

## 3. Data Format

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
