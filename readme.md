# Self-Driving Car Steering Model

## Overview
This project trains a deep learning model to predict steering angles for a self-driving car based on input images from a front-facing camera. The dataset includes images and driving logs, and the model is built using a convolutional neural network (CNN) inspired by NVIDIA's architecture.

## Dataset
The dataset is cloned from the repository:
```
git clone https://github.com/aayushsaini/road_data
```
It contains:
- `driving_log.csv`: CSV file with paths to center, left, and right camera images, along with associated driving data such as steering angle, throttle, and speed.
- `IMG/`: Directory containing the corresponding image files.

## Dependencies
To run this project, install the following dependencies:
```bash
pip install numpy pandas matplotlib keras tensorflow sklearn imgaug opencv-python
```

## Data Preprocessing
The dataset is loaded and preprocessed as follows:
1. **Load CSV Data:** The CSV file is read using Pandas, extracting image paths and steering angles.
2. **Path Cleaning:** Only file names are retained for center, left, and right images.
3. **Data Balancing:** The dataset is balanced by reducing the number of images where the steering angle is near zero.
4. **Train-Validation Split:** The dataset is split into 80% training and 20% validation.
5. **Image Preprocessing:**
   - Cropping unnecessary parts (sky and hood).
   - Converting images to YUV color space.
   - Applying Gaussian blur.
   - Resizing to 200x66 pixels.
   - Normalizing pixel values.

## Model Architecture
The model follows NVIDIA's CNN architecture for self-driving cars:
- 5 Convolutional layers with ELU activation and dropout for regularization.
- Fully connected layers with ELU activation.
- Output layer predicting the steering angle.
- Adam optimizer is used with Mean Squared Error (MSE) loss function.

## Training the Model
The model is trained using:
```python
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid), batch_size=100, shuffle=True)
```
The training progress is visualized by plotting loss curves for training and validation data.

## Saving and Downloading the Model
The trained model is saved as:
```python
model.save('model_a.h5')
```
And can be downloaded from Google Colab using:
```python
from google.colab import files
files.download('model_a.h5')
```

## Results and Evaluation
- The model's performance is evaluated using loss metrics.
- Training and validation loss trends indicate whether the model is overfitting or generalizing well.
- The final model is used for real-time steering angle predictions.

## Usage
1. Clone the dataset repository.
2. Install dependencies.
3. Run the Jupyter Notebook (`model1.ipynb`) to train the model.
4. Save the trained model and use it for inference in self-driving applications.

## Acknowledgments
This project is based on NVIDIA’s deep learning approach for end-to-end self-driving. Special thanks to open-source datasets, Udacity's simulator and libraries enabling research in autonomous driving.

