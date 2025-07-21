# Handwritten Digit Recognition with Convolutional Neural Networks (Keras)

This project implements a Convolutional Neural Network (CNN) using Keras and TensorFlow to classify handwritten digits from the MNIST dataset. The notebook guides you through data loading, preprocessing, model construction, training, evaluation, and visualization of predictions.

## Features

- Loads and preprocesses the MNIST dataset
- Builds a deep CNN with multiple convolutional, batch normalization, ReLU, pooling, and dense layers
- Trains the model and evaluates its performance
- Visualizes sample images and model predictions

## Project Structure

- `Computer Vision Number.ipynb`: Main Jupyter notebook containing all code and explanations

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn

Install dependencies with:

```sh
pip install tensorflow numpy matplotlib seaborn
```

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/v-arnab/handwritten-digit-recognition.git
    cd handwritten-digit-recognition
    ```

2. Open `Computer Vision Number.ipynb` in Jupyter Notebook or VS Code.

3. Run all cells to train and evaluate the CNN model.

## Model Architecture

- Input: 28x28 grayscale images (reshaped to (28, 28, 1))
- 4 convolutional layers (filters: 16, 32, 64, 128; kernel size: 3)
- Batch normalization and ReLU activation after each convolution
- Global average pooling
- Dense output layer (10 units, one per digit class)

## Results

- The model achieves high accuracy on the MNIST test set.
- Visualizations show both sample images and predicted labels.

## License

This project is licensed under the MIT License.