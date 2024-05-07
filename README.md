# Elephant Path Prediction

## Overview

This project focuses on predicting elephant movement trajectories using historical GPS tracking data and a Transformer-based deep learning model.

Elephant movement data consists of GPS coordinates recorded at different timestamps. By learning patterns from historical latitude and longitude sequences, the model predicts the elephant's upcoming locations.

The project uses a Transformer architecture implemented with PyTorch to learn temporal relationships in GPS movement sequences and forecast future positions.

## Project Objectives

- Analyze historical elephant GPS tracking data.
- Preprocess timestamped latitude and longitude information.
- Convert GPS coordinate differences into approximate distance measurements.
- Generate sequential input data for time-series trajectory prediction.
- Develop a Transformer-based model to predict future elephant locations.
- Evaluate predicted trajectories against actual GPS coordinates.
- Visualize historical, actual, and predicted movement paths.

## Dataset

The project uses elephant GPS tracking data from the Kasungu telemetry dataset.

**Dataset file referenced in the notebook:** `Kasungu Telem Pts_June24.csv`

The dataset is loaded with the following fields:

| Feature | Description |
|---|---|
| Index | Record index |
| Tag | Elephant identification tag |
| Latitude | GPS latitude coordinate |
| Longitude | GPS longitude coordinate |
| Time.Stamp | Date and time of GPS observation |

The notebook loads this CSV file from Google Drive. The dataset is not included in the repository, so it must be obtained separately and the file path updated before execution.

## Technologies and Libraries

- **Python** – Core programming language.
- **PyTorch** – Transformer architecture, model training, and inference.
- **Pandas** – Dataset loading, sorting, and preprocessing.
- **NumPy** – Numerical operations and sequence generation.
- **Scikit-learn** – Data splitting, normalization, and evaluation metrics.
- **Matplotlib** – Training curves and trajectory visualization.
- **GeoPandas** – Geographic data processing.
- **Contextily** – Geographic map visualization support.
- **Shapely** – Geometric objects and spatial operations.
- **Geopy** – Geographic distance calculations.

## Methodology

### 1. Data Loading and Preprocessing

The GPS tracking dataset is loaded using Pandas.

The preprocessing steps include:

- Selecting the required columns.
- Converting timestamp values into datetime format.
- Sorting observations by elephant tag and timestamp.
- Calculating time differences between consecutive observations.
- Calculating latitude and longitude differences.
- Converting coordinate differences into approximate distances in meters.

This prepares the historical tracking information for sequential modeling.

### 2. GPS Coordinate Conversion

GPS latitude and longitude differences are converted into approximate distances in meters.

The conversion accounts for the reference latitude when estimating longitudinal distance.

This helps represent geographical movement in terms of physical distance rather than relying only on coordinate differences.

### 3. Sequence Generation

The project uses a sliding-window approach to generate input and target sequences.

The implemented configuration uses:

- Historical sequence length: 15 observations.
- Prediction length: 3 future observations.
- Input features: Latitude and Longitude.
- Output features: Three future latitude-longitude coordinate pairs.

The model receives a sequence of historical GPS coordinates and predicts the next three locations.

Elephant tracking records are processed separately using their respective identification tags. Elephants with insufficient tracking observations are excluded from sequence generation.

### 4. Training and Testing Data

The generated sequences are divided into training and testing datasets.

The notebook uses an 80:20 split and creates PyTorch TensorDatasets and DataLoaders for batch-based model training and evaluation.

The DataLoader uses a batch size of 16.

### 5. Transformer Architecture

A Transformer model is implemented using PyTorch.

The architecture includes:

- Linear embedding layer to project GPS coordinates into the model dimension.
- Sinusoidal positional encoding to represent sequence positions.
- Transformer encoder-decoder layers.
- Fully connected output layer for future coordinate prediction.

**Model configuration used in the notebook:**

| Parameter | Value |
|---|---:|
| Input dimension | 2 |
| Model dimension (`d_model`) | 64 |
| Attention heads | 4 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Feed-forward dimension | 128 |
| Output dimension | 6 |
| Maximum sequence length | 200 |

The output dimension is six because the model predicts three future GPS locations, with two coordinates (latitude and longitude) per location.

The model automatically uses a CUDA-enabled GPU when available; otherwise, it runs on the CPU.

### 6. Model Training

The Transformer is trained using historical GPS sequences and their corresponding future coordinate values.

Training and validation losses are monitored across epochs. Root Mean Squared Error (RMSE) is also calculated to assess prediction performance.

The trained model is saved as `Transformer.pth` and loaded during evaluation.

### 7. Model Evaluation

The model's predicted GPS coordinates are compared with the actual future GPS coordinates.

The notebook evaluates performance using:

- **RMSE (Root Mean Squared Error):** Measures the prediction error between predicted and actual coordinates.
- **Average Displacement Error (ADE):** Measures the average distance between predicted and actual positions across the predicted trajectory.
- **Final Displacement Error (FDE):** Measures the distance between the final predicted location and the actual final location.
- **Mean Angle Error:** Measures the difference in movement direction between predicted and actual trajectories.

Additional geographic distance calculations are performed to express trajectory errors in meters.

### 8. Trajectory Visualization

The project visualizes the predicted trajectories against the actual elephant movement paths.

The visualizations include:

- Historical GPS trajectory.
- Actual future trajectory.
- Transformer-predicted future trajectory.
- Best-performing prediction examples.
- Worst-performing prediction examples.
- Trajectory comparisons represented in meters.

These visualizations help understand how closely the predicted movement paths follow the actual elephant movement patterns.

## Execution Environment

The notebook was developed using Google Colab and includes GPU support.

To execute the project:

1. Open `Transformer.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required libraries.
3. Provide access to the elephant GPS tracking CSV dataset.
4. Update the dataset path in the notebook.
5. Execute the preprocessing and sequence-generation steps.
6. Train the Transformer model.
7. Evaluate the model and visualize the predicted trajectories.

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib torch scikit-learn
pip install geopandas contextily shapely geopy
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/indhu68/Elephant_Path_Prediction.git
```

Navigate to the project directory:

```bash
cd Elephant_Path_Prediction
```

Open the notebook:

```bash
jupyter notebook Transformer.ipynb
```

Before running the notebook, make sure the dataset is available and update its path accordingly.

The saved model file `Transformer.pth` is also required when running the evaluation section independently. It is generated during model training.

## Key Learning Outcomes

- Working with real-world GPS tracking and time-series data.
- Processing timestamped geographical observations.
- Converting latitude and longitude differences into distance measurements.
- Generating sliding-window sequences for trajectory forecasting.
- Implementing Transformer architectures using PyTorch.
- Understanding positional encoding and multi-head attention.
- Training deep learning models for sequential prediction.
- Evaluating trajectory predictions using RMSE, ADE, FDE, and angle error.
- Visualizing predicted and actual geographical movement paths.

## Conclusion

This project explores the application of Transformer-based deep learning for elephant movement trajectory prediction.

By learning from historical GPS tracking sequences, the model predicts future latitude and longitude coordinates. The project combines time-series preprocessing, Transformer architecture implementation, model evaluation, and trajectory visualization to study elephant movement patterns.

The approach demonstrates how deep learning and GPS-based trajectory forecasting can be applied to wildlife movement analysis and can serve as a foundation for further research in wildlife monitoring and conservation.
