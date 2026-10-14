# Handwritten Digit Recognition (MNIST)

**InfozaTech Machine Learning Internship**

A deep learning project that recognises handwritten digits (0-9) from the MNIST dataset. Two models are trained and compared: a fully connected neural network and a convolutional neural network (CNN).

## Dataset

- **MNIST**: 70,000 grayscale images of handwritten digits, each 28x28 pixels
- 60,000 training images and 10,000 test images
- 10 classes (digits 0 to 9)
- Loaded directly through `keras.datasets.mnist`

## Preprocessing

- Pixel values scaled from 0-255 to 0-1
- Added a channel dimension, giving an input shape of (28, 28, 1)
- 10% of the training data held out for validation
- Fixed random seed (42) for reproducible results

## Models

| Model | Parameters | Test accuracy | Test loss |
|-------|-----------|---------------|-----------|
| dense_nn | 109,386 | 97.53% | 0.0811 |
| cnn | 225,034 | **99.22%** | 0.0250 |

**Dense NN:** Flatten, Dense(128, ReLU), Dropout(0.2), Dense(64, ReLU), Dense(10, Softmax)

**CNN:** Conv2D(32), MaxPooling, Conv2D(64), MaxPooling, Flatten, Dropout(0.3), Dense(128, ReLU), Dense(10, Softmax)

Both models use the Adam optimizer, sparse categorical cross-entropy loss, a batch size of 128 and 8 epochs.

## Results

The CNN performed best, reaching **99.22% test accuracy** and misclassifying only 78 of 10,000 test images, compared with 97.53% for the dense network. The CNN learns spatial patterns such as edges and curves, which a flattened dense network cannot capture as well.

Per-class F1 scores were all above 0.986. Digit 9 had the lowest recall (97.6%). It was most often mistaken for 4 (8 times) and 5 (5 times), which look similar when handwritten. Other common errors were 2 predicted as 7 (6 times) and 3 predicted as 5 (5 times). The 9-to-4 confusion was mostly one-directional: only 1 true 4 was predicted as 9.

### Visual outputs

| File | Description |
|------|-------------|
| `sample_digits.png` | Example images from the dataset |
| `training_curves.png` | Validation accuracy and loss per epoch |
| `confusion_matrix.png` | Test-set confusion matrix of the CNN |
| `predictions.png` | Random test predictions with confidence |
| `misclassified.png` | Examples the model got wrong |

## Project structure

```
Hand_Written_Digits_Recognition/
|-- digit_recognition.py
|-- model_comparison.csv
|-- digit_model.keras
|-- requirements.txt
|-- sample_digits.png
|-- training_curves.png
|-- confusion_matrix.png
|-- predictions.png
|-- misclassified.png
|-- README.md
```

## How to run

```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python digit_recognition.py
```

Python 3.10-3.12 is recommended. An internet connection is needed on the first run to download MNIST. Training takes a few minutes on CPU.

## Tech stack

Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn

## Author

Sona Rajendren | MCA | Presidency University
