# Rice Plant Disease Classification

## Project Overview

This project is aimed at developing a model for classifying three major diseases that attack rice plants: leaf blast, bacterial blight, and brown spot. The main objectives are to prepare a comprehensive data analysis report, create a classification model, and analyze data augmentation techniques.

## Problem Statement

The goal of this project is to classify rice plant diseases accurately, which is crucial for early disease detection and crop management. The specific tasks include:

- Task 1: Data Analysis Report
- Task 2: Disease Classification Model
- Task 3: Data Augmentation Analysis

## Project Structure

- **Task 1: Data Analysis Report**
  - The dataset contains 119 images of rice plants affected by Bacterial leaf blight, Brown spot, and Leaf smut.
  - Images have been preprocessed and resized to 256x256 pixels.
  
- **Task 2: Disease Classification Model**
  - Model Architecture
    - A Convolutional Neural Network (CNN) is used for image classification.
    - Preprocessing layers include resizing images to 256x256 pixels, rescaling pixel values to [0, 1], and data augmentation techniques like random flips and rotations.
    - Convolutional layers, dropout layers to prevent overfitting, and a flattening layer for 1D data transformation.
    - Dense layers with ReLU activation and an output layer with softmax activation for multi-class classification.
  - Model Training
    - Trained using the Adam optimizer and sparse categorical cross-entropy loss.
    - Training for 50 epochs with a batch size of 4.
  
- **Task 3: Data Augmentation Analysis**
  - Introduced dropout layers to mitigate overfitting.
  - Improved generalization on both training and validation data.

## Results

Based on the project's findings:

- The initial model exhibited overfitting issues.
- With training, the model started to generalize better after introducing dropout layers.
- The final model achieved an accuracy of approximately 90.91% on the training dataset and 87.5% on the test dataset.
- Further improvements can be achieved by fine-tuning hyperparameters, exploring different architectures, and leveraging advanced techniques like transfer learning.

## Conclusion

This project demonstrates the process of developing an image classification model for detecting diseases in rice plants. By analyzing the data, creating an appropriate model architecture, and applying data augmentation techniques, we built a model that showed promise in classifying the different diseases. Further optimization and experimentation can lead to even better results and a more robust classification solution.

## How to Use

Provide instructions on how to use your project. For example:
- Clone this repository to your local machine.
- Install the required dependencies (mention them).
- Run `model_training.py` to train the model.
- Use `model_evaluation.py` to evaluate the model's performance.
