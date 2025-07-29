# MNIST Digit Classification with CNN

A Convolutional Neural Network implementation for handwritten digit recognition using the MNIST dataset. This project demonstrates the power of CNNs for image classification tasks.

## Overview

This project implements a CNN model to classify handwritten digits (0-9) from the MNIST dataset. The model achieves high accuracy through convolutional layers, pooling, and dense layers, showcasing fundamental deep learning concepts for computer vision.

## Features

- **Data Visualization**: Display sample MNIST digits with labels
- **CNN Architecture**: Custom convolutional neural network with:
  - Conv2D layer (32 filters, 3x3 kernel)
  - MaxPooling2D layer (2x2 pool size)
  - Flatten and Dense layers
  - Softmax output for 10-class classification
- **Model Training**: 20 epochs with validation monitoring
- **Performance Evaluation**: Test accuracy and loss metrics
- **Predictions**: Real-time digit prediction on test images
- **Custom Image Testing**: Load and predict custom digit images

## Dataset

The MNIST dataset contains:

- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image size**: 28x28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

## Model Architecture

```
Sequential Model:
├── Conv2D(32, (3,3), activation='relu') - Input: (28,28,1)
├── MaxPooling2D((2,2))
├── Flatten()
├── Dense(32, activation='relu')
└── Dense(10, activation='softmax')
```

## Requirements

```python
numpy
pandas
matplotlib
seaborn
tensorflow
keras
scikit-learn
pillow
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/[username]/mnist-cnn-classifier.git
cd mnist-cnn-classifier
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn pillow
```

## Usage

1. **Run the Jupyter notebook**:

```bash
jupyter notebook Day5_Convolutional.ipynb
```

2. **Key sections**:
   - Data loading and preprocessing
   - Visualization of sample digits
   - Model creation and compilation
   - Training with performance tracking
   - Evaluation and predictions
   - Custom image testing

## Results

- **Training Time**: Approximately 20 epochs
- **Test Accuracy**: High accuracy on MNIST test set
- **Confusion Matrix**: Detailed classification performance
- **Custom Predictions**: Supports external digit image classification

## Key Concepts Demonstrated

- **Convolutional Neural Networks**: Feature extraction through convolution
- **Data Preprocessing**: Normalization and one-hot encoding
- **Model Evaluation**: Loss, accuracy, and confusion matrix analysis
- **Image Processing**: Custom image loading and preprocessing
- **Deep Learning Pipeline**: End-to-end ML workflow

## File Structure

```
├── Day5_Convolutional.ipynb    # Main notebook with CNN implementation
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies (optional)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- MNIST dataset from Yann LeCun's website
- TensorFlow/Keras for deep learning framework
- Matplotlib/Seaborn for data visualization

---

**Note**: This project is part of a deep learning series focusing on convolutional neural networks for computer vision tasks.
