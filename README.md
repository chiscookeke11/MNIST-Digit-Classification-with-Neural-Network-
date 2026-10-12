# MNIST Digit Classification with Neural Network

A handwritten digit classification project built with Python, TensorFlow, and Keras. The project uses a neural network to recognize handwritten digits from 0 to 9 using the MNIST dataset. It covers data loading, preprocessing, neural network development, model training, evaluation, visualization, and testing with a custom image.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How the Project Works](#how-the-project-works)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Model Architecture](#model-architecture)
- [Model Evaluation](#model-evaluation)
- [Testing with a Custom Image](#testing-with-a-custom-image)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Challenges and Limitations](#challenges-and-limitations)
- [Future Improvements](#future-improvements)
- [Key Lessons Learned](#key-lessons-learned)
- [Author](#author)

## Project Overview

Handwritten digit recognition is a classic computer vision and machine learning problem. Although humans can easily recognize most handwritten digits, teaching a computer to identify them requires converting images into numerical data and training a model to learn patterns from examples.

In this project, I built a feedforward neural network using TensorFlow and Keras. The model learns from the MNIST dataset, which contains images of handwritten digits, and predicts which digit is represented by an input image.

The project also explores how to evaluate a trained model using test data, accuracy, predictions, and a confusion matrix. Finally, it includes an experiment with preprocessing a custom handwritten image for prediction.

### Project Highlights

- Loaded and explored the MNIST handwritten digit dataset.
- Prepared image data for neural network training.
- Built a neural network using Keras Sequential API.
- Trained the model for 10 epochs.
- Evaluated performance on unseen test data.
- Generated predictions for test images.
- Visualized classification performance with a confusion matrix.
- Experimented with predicting digits from a custom image.

## Objectives

The primary objectives of this project are to:

1. Understand how image classification works.
2. Learn how image data is represented as numerical arrays.
3. Build a neural network using TensorFlow and Keras.
4. Understand the relationship between input features, hidden layers, and output classes.
5. Train a model to recognize handwritten digits.
6. Evaluate model performance using accuracy and a confusion matrix.
7. Apply image preprocessing techniques to a real-world handwritten image.
8. Understand the challenges of using a model trained on one dataset to predict images from a different source.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Machine learning and neural network framework |
| Keras | Building, compiling, training, and evaluating the model |
| NumPy | Numerical arrays and data manipulation |
| Matplotlib | Visualizing images and results |
| Seaborn | Plotting the confusion matrix |
| OpenCV | Image processing and preprocessing |
| Pillow (PIL) | Image loading and manipulation |
| Jupyter Notebook | Interactive development and experimentation |

## Dataset

### The MNIST Dataset

The project uses the MNIST dataset provided by Keras.

MNIST contains grayscale images of handwritten digits ranging from 0 to 9.

Each image has a resolution of **28 × 28 pixels**. Each pixel contains an intensity value representing how bright or dark that pixel is.

The dataset contains 70,000 images divided into two groups:

| Dataset | Number of images | Purpose |
|---|---:|---|
| Training set | 60,000 | Learning patterns from examples |
| Test set | 10,000 | Evaluating performance on unseen examples |

Each image has a corresponding label indicating the correct digit.

For example, an image of a handwritten `7` has the label `7`.

### Loading the Dataset

The dataset is loaded using Keras:

```python
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

The resulting arrays have these shapes:

```text
X_train: (60000, 28, 28)
Y_train: (60000,)
X_test:  (10000, 28, 28)
Y_test:  (10000,)
```

Here:

- `X_train` contains the training images.
- `Y_train` contains their corresponding labels.
- `X_test` contains the test images.
- `Y_test` contains the correct labels for the test images.

## Project Structure

The repository contains the following main files:

```text
MNIST-Digit-Classification-with-Neural-Network/
│
├── MNIST_digit_classification_with_Neural_Network.ipynb
├── test_image_grayscale.jpg
├── test_image_resized.jpg
├── README.md
└── sample_data/
```

- `MNIST_digit_classification_with_Neural_Network.ipynb` — The main notebook containing the implementation, training, evaluation, visualizations, and custom-image experiment.
- `test_image_grayscale.jpg` — A grayscale version of a test image.
- `test_image_resized.jpg` — A resized test image.
- `sample_data/` — A directory included in the repository for supporting sample files and experimentation.
- `README.md` — Documentation for the project.

*Note:* Image paths used in the notebook should match the actual locations and filenames in your local repository.

## How the Project Works

The workflow follows these stages:

```text
MNIST Dataset
      |
      v
Load and Explore Images
      |
      v
Preprocess Image Data
      |
      v
Build Neural Network
      |
      v
Compile the Model
      |
      v
Train for 10 Epochs
      |
      v
Evaluate on Test Data
      |
      v
Generate Predictions
      |
      v
Analyze Confusion Matrix
      |
      v
Experiment with Custom Images
```

Each stage contributes to the development and evaluation of the digit classification system.

## Step-by-Step Implementation

### Step 1: Import the Required Libraries

The first step is importing the libraries needed for data manipulation, visualization, model development, and image processing.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

tf.random.set_seed(3)
```

**What these libraries do:**

- NumPy handles numerical arrays.
- Matplotlib displays images and plots.
- Seaborn visualizes the confusion matrix.
- OpenCV provides image processing operations.
- TensorFlow and Keras provide the neural network tools.
- The random seed helps make certain random operations more reproducible.

### Step 2: Load the MNIST Dataset

The next step is loading the dataset.

```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

Keras downloads the dataset the first time it is needed and typically caches it for later use.

The dataset is divided into training and test sets so the model can be evaluated on images it did not train on.

### Step 3: Explore the Dataset

Before training, it is useful to understand the dimensions and structure of the data.

```python
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
```

You can also inspect a single image:

```python
print(X_train[10])
print(X_train[10].shape)
```

Each image is represented by a two-dimensional array containing 28 rows and 28 columns of pixel values.

Visualizing the image makes the numerical representation easier to understand:

```python
plt.imshow(X_train[10], cmap="gray")
plt.title(f"Digit: {Y_train[10]}")
plt.axis("off")
plt.show()
```

This step demonstrates how an image becomes numerical input that a machine learning model can process.

### Step 4: Understand the Pixel Values

The MNIST images use grayscale pixel values, typically ranging from 0 to 255.

- `0` represents black.
- `255` represents white.
- Values between 0 and 255 represent different shades of gray.

A neural network generally trains more conveniently when the input values are scaled to a smaller numerical range.

### Step 5: Normalize the Image Data

The training and test images are normalized by dividing their pixel values by 255.

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

After normalization, the pixel values range from 0 to 1.

For example:

```text
Original pixel value: 128
Normalized value:     128 / 255 ≈ 0.502
```

Normalization helps keep input values within a consistent range and can improve training behavior.

**Important:** Apply the same normalization convention to custom images when making predictions.

### Step 6: Build the Neural Network

The model is built using Keras Sequential API.

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(10, activation="sigmoid"),
])
```

The network contains four layers.

**1. Flatten layer**

```python
keras.layers.Flatten(input_shape=(28, 28))
```

Each 28 × 28 image is converted into a one-dimensional vector of 784 values.

\[
28 \times 28 = 784
\]

This makes the image suitable for the following dense layers.

**2. First hidden layer**

```python
keras.layers.Dense(50, activation="relu")
```

This layer contains 50 neurons. It learns patterns and combinations of input features.

The ReLU activation function introduces nonlinearity into the model.

**3. Second hidden layer**

```python
keras.layers.Dense(50, activation="relu")
```

The second hidden layer learns additional patterns that help distinguish between digits.

**4. Output layer**

```python
keras.layers.Dense(10, activation="sigmoid")
```

The output layer contains 10 units, corresponding to the 10 possible digits: 0 through 9.

The model produces 10 output scores for each input image. The notebook uses `argmax` to select the class with the highest score.

### Step 7: Compile the Model

Before training, the model is compiled.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

The compilation step configures the training process.

- **Adam optimizer:** Updates the model's weights during training.
- **Sparse categorical cross-entropy:** Calculates the classification loss when the labels are represented as integer class IDs, such as 0, 1, and 2.
- **Accuracy:** Measures the proportion of predictions that match the correct labels.

The labels do not need to be one-hot encoded for this loss function.

### Step 8: Train the Neural Network

The model is trained using the training images and their labels.

```python
model.fit(
    X_train,
    Y_train,
    epochs=10
)
```

The model trains for 10 epochs.

An epoch is one complete pass through the training dataset.

During training, the network:

1. Receives a batch of training images.
2. Produces predictions.
3. Calculates the loss by comparing predictions with the correct labels.
4. Updates its weights using the optimizer.
5. Repeats the process until the training data has been processed for the epoch.

This process is repeated for the specified number of epochs so the model can learn useful patterns.

### Step 9: Evaluate the Model

After training, the model is evaluated using the test dataset.

```python
loss, accuracy = model.evaluate(X_test, Y_test)

print("Test loss:", loss)
print("Test accuracy:", accuracy)
```

The test dataset contains 10,000 images that were not used for training.

Evaluation provides an indication of how well the model generalizes to unseen examples.

The notebook records approximately:

- Training accuracy: 94.26%
- Test accuracy: 94.20%

These figures are the results recorded in the notebook, rather than a guarantee that every run or environment will produce exactly the same values.

### Step 10: Generate Predictions

Once the model has been trained, it can predict the digits represented by test images.

```python
Y_pred = model.predict(X_test)
```

The prediction output contains 10 scores for each image, one for each possible digit.

To obtain the predicted digit, the index of the highest score is selected.

```python
label_for_first_image = np.argmax(Y_pred[0])

print("Predicted digit:", label_for_first_image)
print("Actual digit:", Y_test[0])
```

For the first test image, comparing the predicted digit with the actual label helps determine whether the model classified it correctly.

### Step 11: Convert All Predictions to Class Labels

The predicted scores can be converted into class labels for all test images.

```python
Y_pred_labels = [
    np.argmax(prediction)
    for prediction in Y_pred
]
```

Each entry in `Y_pred_labels` represents the model's predicted digit for a corresponding test image.

These predicted labels can now be compared with the true labels to analyze classification performance.

### Step 12: Build a Confusion Matrix

A confusion matrix provides a more detailed view of the model's predictions.

```python
conf_mat = confusion_matrix(
    Y_test,
    Y_pred_labels
)
```

A confusion matrix compares the actual classes with the predicted classes.

For a 10-class digit classification problem, the matrix has 10 rows and 10 columns.

- Rows represent the true digits.
- Columns represent the predicted digits.
- Diagonal entries represent correct classifications.
- Off-diagonal entries represent misclassifications.

For example, a value in row `3`, column `5` indicates how many actual instances of digit 3 were predicted as digit 5, using zero-based class indexing.

### Step 13: Visualize the Confusion Matrix

The confusion matrix can be displayed using Seaborn.

```python
plt.figure(figsize=(15, 7))

sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.title("MNIST Confusion Matrix")
plt.show()
```

This visualization helps identify which digits the model classifies accurately and which digits it tends to confuse.

For example, handwritten digits with similar shapes may sometimes be misclassified as one another.

## Testing with a Custom Image

In addition to evaluating MNIST test images, the project experiments with predicting a digit from an external handwritten image.

This step introduces a practical challenge: an image captured with a camera does not necessarily resemble the clean, centered images in MNIST.

### Step 14: Load the Custom Image

OpenCV can be used to load the image.

```python
photo = cv2.imread("test_image_grayscale.jpg")

if photo is None:
    raise FileNotFoundError(
        "Could not load the image. Check the file path."
    )
```

The image path should point to the actual file in the project directory.

If the file is already grayscale, use `cv2.IMREAD_GRAYSCALE` when loading it instead.

### Step 15: Convert to Grayscale

If the image is in color, convert it to grayscale.

```python
gray = cv2.cvtColor(
    photo,
    cv2.COLOR_BGR2GRAY
)
```

Grayscale conversion reduces the image to one intensity value per pixel, which is appropriate for a model trained on grayscale images.

Skip this conversion if the image has already been loaded in grayscale.

### Step 16: Apply Thresholding

Thresholding separates the foreground digit from the background.

```python
gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

plt.imshow(binary, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```

The Gaussian blur reduces small amounts of noise. Otsu's thresholding method automatically selects a threshold value, while the inverse binary option makes darker foreground marks appear white and the lighter background appear black.

Inspect the result to confirm that the digit is visible and the background is mostly black.

### Step 17: Prepare the Digit for MNIST

MNIST images are 28 × 28 pixels, so the custom image needs to be cropped and resized.

The preprocessing workflow should:

1. Identify the foreground pixels.
2. Crop around the digit.
3. Resize the digit while preserving its aspect ratio.
4. Center it in a 28 × 28 canvas.
5. Normalize the pixel values to the range 0–1.
6. Add the batch dimension required by the model.

For example, the final prepared image should have a shape of `(1, 28, 28)` for the current model.

The exact cropping and centering procedure is important because an incorrectly prepared image can lead to inaccurate predictions even when the model performs well on MNIST.

### Step 18: Predict the Custom Digit

After preprocessing, pass the image to the trained model.

```python
digit_img = prepare_digit(binary)

plt.imshow(digit_img, cmap="gray")
plt.title("Prepared 28 × 28 Image")
plt.axis("off")
plt.show()

input_img = digit_img.reshape(1, 28, 28)

prediction = model.predict(input_img)

predicted_digit = np.argmax(prediction[0])

print("Predicted digit:", predicted_digit)
```

The `prepare_digit()` function must be defined before running this example. Its output should be a normalized 28 × 28 array with a black background and a bright foreground digit.

If the external image contains multiple digits, each digit should ideally be detected and processed separately because this model predicts one digit per image.

## Model Architecture

The network used in the notebook is a fully connected feedforward neural network.

| Layer | Configuration | Purpose |
|---|---|---|
| Input and flattening | 28 × 28 → 784 values | Converts each image into a vector |
| Hidden layer 1 | 50 neurons, ReLU | Learns image patterns |
| Hidden layer 2 | 50 neurons, ReLU | Learns additional feature combinations |
| Output layer | 10 neurons, sigmoid | Produces scores for the ten digit classes |

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss for 10 epochs.

**Implementation note:** The current notebook uses sigmoid in its output layer. For a conventional single-label, 10-class classifier, a softmax output layer is commonly used with categorical cross-entropy or sparse categorical cross-entropy. That is a possible future improvement, not a change to the reported implementation.

## Model Evaluation

The notebook records a training accuracy of approximately 94.26% and a test accuracy of approximately 94.20%.

The similarity between these figures suggests that the model performs comparably on the training and test datasets in the recorded run. However, accuracy alone does not explain every type of classification error.

The confusion matrix provides additional insight into which digits are predicted correctly and which are confused with other classes.

Custom-image testing also highlights an important distinction: performance on MNIST test images does not guarantee the same performance on photographs of handwritten digits. Lighting, image orientation, background texture, digit thickness, cropping, and centering can all affect predictions.

## Installation and Setup

### Prerequisites

Install the following before running the notebook:

- Python 3.10 or another Python version compatible with your chosen TensorFlow release.
- Git.
- Visual Studio Code.
- The Jupyter extension for VS Code.

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/chiscookeke11/MNIST-Digit-Classification-with-Neural-Network-.git
```

Navigate into the project directory:

```bash
cd MNIST-Digit-Classification-with-Neural-Network-
```

### Step 2: Create a Virtual Environment

Using Python's built-in virtual environment support:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

If you are using Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Step 3: Install the Dependencies

Install the packages used by the notebook:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow numpy matplotlib seaborn opencv-python pillow jupyter
```

The exact TensorFlow version available depends on your Python version and operating system.

### Step 4: Open the Project in VS Code

From the project directory, run:

```bash
code .
```

Open `MNIST_digit_classification_with_Neural_Network.ipynb`.

### Step 5: Select the Python Kernel

In VS Code:

1. Open the notebook.
2. Select **Select Kernel**.
3. Choose the Python interpreter from your `.venv` environment.
4. Run the notebook cells in order.

The first dataset-loading step may download MNIST if it is not already cached locally.

## Running the Project

To reproduce the experiment:

1. Install the required packages.
2. Open the notebook in VS Code.
3. Load the MNIST dataset.
4. Explore the images and labels.
5. Normalize the image arrays.
6. Build and compile the neural network.
7. Train the model for 10 epochs.
8. Evaluate the model using the test dataset.
9. Generate predictions for test images.
10. Create and visualize the confusion matrix.
11. Run the custom-image experiment after preparing a suitable input image.

### Hardware Notes

The model can run on a CPU, although training speed depends on your hardware.

Standard TensorFlow 2.11 and later does not provide CUDA GPU support for native Windows installations. A compatible NVIDIA GPU can be used through a properly configured WSL2 environment, while Google Colab is another option for cloud-based GPU execution.

GPU acceleration is not required to complete this beginner-level MNIST project.

## Challenges and Limitations

### 1. Custom Image Preprocessing

Photographs are different from the images used during MNIST training. The model expects a centered 28 × 28 grayscale digit, so the external image needs appropriate cropping, thresholding, resizing, and normalization.

### 2. Multiple Digits in One Image

The current model predicts a single digit per image. A photograph containing a sequence such as `24` must be segmented into individual digits before each can be classified independently.

### 3. Background Noise

Notebook lines, shadows, uneven lighting, and camera perspective can affect the extracted digit. Poor preprocessing may make the input substantially different from the MNIST training examples.

### 4. Classification Errors

The model can confuse digits that have similar handwritten shapes. The confusion matrix helps investigate these errors.

### 5. Limited Generalization

The model is trained on MNIST, so it may not perform consistently on every handwriting style or image-capture condition.

## Future Improvements

Possible improvements to this project include:

- Experimenting with a softmax output layer for mutually exclusive digit classes.
- Comparing the current dense neural network with a convolutional neural network (CNN).
- Saving the trained model so it can be loaded without retraining.
- Improving custom-image preprocessing and automatic digit segmentation.
- Adding confidence scores and displaying the top predicted classes.
- Testing the model on a wider variety of handwritten images.
- Building a simple web interface where users can upload images or draw digits.
- Tracking precision, recall, and per-class performance in addition to accuracy.
- Adding a dedicated test script for evaluating custom images.

## Key Lessons Learned

Through this project, I explored several foundational machine learning concepts:

- Loading datasets with TensorFlow and Keras.
- Understanding image arrays and pixel values.
- Normalizing numerical data.
- Building neural networks using dense layers.
- Understanding activation functions, optimizers, and loss functions.
- Training models over multiple epochs.
- Evaluating models on unseen test data.
- Interpreting prediction scores and class labels.
- Using a confusion matrix to understand classification errors.
- Applying computer vision techniques to prepare external images for inference.
- Understanding why training data and real-world inputs need consistent preprocessing.

## Conclusion

This project demonstrates an end-to-end introductory machine learning workflow for handwritten digit classification. It combines dataset exploration, preprocessing, neural network development, model training, evaluation, and an experiment with external images.

Beyond recognizing digits, the project provides a foundation for understanding how neural networks learn from image data and how preprocessing and evaluation influence the reliability of machine learning predictions.

## Author

**Chinedu Okeke Emmanuel**

Software Engineer | AI/ML Engineer | Open Source Contributor

- GitHub: [@chiscookeke11](https://github.com/chiscookeke11)
- Project Repository: [MNIST Digit Classification with Neural Network](https://github.com/chiscookeke11/MNIST-Digit-Classification-with-Neural-Network-)

---

If you find this project useful for learning machine learning, feel free to explore the notebook, experiment with the architecture, and build upon the implementation.
