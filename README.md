# Vision Model Suite

This project provides a simple framework to benchmark various open-source vision models for performance and output quality. It is designed to be easily extensible with new models.

##  Models Included

*   **Apple FastVLM**: `apple_fastvlm.py`
*   **LlavaOneVision**: `llava_one_vision.py`
*   **Moondream2**: `moondream2.py`
*   **OpenGVLab InternVL**: `opengvlab.py`

## Prerequisites

*   Python 3.8+
*   NVIDIA GPU with CUDA installed (recommended for performance)
*   Git

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jebin2/vision.git
    cd vision
    ```

2.  **Install the required Python packages:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

To run the benchmark, you need a sample image in the root directory of the project.

1.  **Add a test image:**
    Place an image named `test.png` in the root of the project folder.

2.  **Run the benchmark script:**
    The script will automatically discover all vision models, then load and run them one by one to measure performance.
    ```bash
    python benchmark.py
    ```

The script will print the loading time, generation time, and the generated text for each model before cleaning up its memory and moving to the next.

## Adding a New Model

1.  Create a new Python file (e.g., `my_new_model.py`).
2.  Inside the file, create a class that inherits from `VisionModel`.
3.  Implement the required `load_model()` and `generate()` methods.
4.  The `benchmark.py` script will automatically discover and include your new model in its next run.