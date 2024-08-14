# Mood Detection Using Camera

This project implements a machine learning model to detect the mood of a person based on their facial expressions using a camera. The tool captures real-time video feed and classifies the mood into predefined categories.

## File Structure

- `data/`: Contains raw and processed data.
  - `raw/`: Directory for storing raw datasets.
  - `processed/`: Directory for storing processed data.
- `models/`: Includes model architecture, training, and inference scripts.
  - `model.py`: Defines the model architecture.
  - `inference.py`: Handles model loading and prediction.
- `utils/`: Contains utility scripts for various tasks.
  - `camera_capture.py`: Captures video feed and performs mood prediction in real-time.
  - `data_preprocessing.py`: Preprocesses data for training.
  - `preprocess_images.py`: Additional script for preprocessing images before training.
- `train.py`: Script to train the model.
- `main.py`: Main script to run the application.
- `requirements.txt`: Lists the Python packages required for the project.
- `README.md`: Project documentation.

## Installation

1. **Create a virtual environment** and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Data**:
   - Place your dataset in the `data/raw/` directory.
   - Run data preprocessing to prepare the dataset:
     ```bash
     python utils/data_preprocessing.py
     ```
   - Optionally, use `preprocess_images.py` to further preprocess images before training:
     ```bash
     python utils/preprocess_images.py
     ```

2. **Train Model**:
   - Configure the model architecture in `models/model.py`.
   - Train the model:
     ```bash
     python train.py
     ```

3. **Run Inference**:
   - Capture video feed and perform mood detection:
     ```bash
     python main.py
     ```

## Notes

- Ensure you have a working camera and proper drivers installed for capturing video feed.
- The model needs to be trained before running inference.
- You may need to adjust model parameters and hyperparameters based on your dataset.


