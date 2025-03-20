# Defungi

Defungi is a deep learning-based project aimed at classifying images of fungal infections. Using Convolutional Neural Networks (CNNs), the model is trained on the **Defungi dataset** containing images classified into different fungal categories: **H1**, **H2**, **H3**, **H5**, and **H6**. This project demonstrates the application of CNNs for image classification in the field of medical diagnostics.

## Features

- **Image Classification**: The model classifies fungal infection images into the following categories:
  - **H1**
  - **H2**
  - **H3**
  - **H5**
  - **H6**

- **Deep Learning Model**: Built using Convolutional Neural Networks (CNNs) for high accuracy and robustness in identifying different types of fungi from images.
  
- **Training & Evaluation**: The project includes a pipeline for training and evaluating the model with metrics like accuracy, precision, recall, and F1-score.

## Dataset

The **Defungi dataset** is a collection of labeled images of fungal infections. The dataset is divided into five classes: **H1**, **H2**, **H3**, **H5**, and **H6**.

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras for building and training the CNN model.
- **Data Preprocessing**: OpenCV and PIL for image processing and augmentation.
- **Model**: Convolutional Neural Network (CNN) architecture for image classification.
- **Python**: The primary programming language for model implementation and training.

## Installation

To get started with Defungi, follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/shashiawari/defungi.git
   ```

2. Install the required dependencies:

   ```bash
   cd defungi
   pip install -r requirements.txt
   ```

3. Download the **Defungi dataset** and place it in the `data/` folder.

4. Start training the model:

   ```bash
   python train.py
   ```

5. After training, you can evaluate the model using:

   ```bash
   python evaluate.py
   ```

6. The trained model will be saved to the `model/` directory.

## Model Architecture

The CNN model used in this project consists of multiple convolutional layers, pooling layers, and fully connected layers designed to classify images of fungal infections into the appropriate classes (H1, H2, H3, H5, H6). The architecture includes:

- Input layer: Takes in preprocessed images.
- Convolutional layers: Extract features from images.
- Pooling layers: Reduce spatial dimensions.
- Dense layers: Classify the images into one of the five classes.

## Results

After training the model on the **Defungi dataset**, the CNN achieved the following performance metrics:

- **Accuracy**: [70%]

## Contributing

We welcome contributions to improve Defungi! If you'd like to contribute, please fork the repository and create a pull request. Before submitting, make sure to:

- Test your changes thoroughly.
- Follow the coding standards and best practices.

