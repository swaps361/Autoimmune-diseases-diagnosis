# Quantitative Analysis of Histopathological Images for Autoimmune Diseases Diagnosis

## Description
1. This project focuses on the quantitative analysis of histopathological images to diagnose autoimmune diseases, specifically Lupus, Arthritis, and Sclerosis. 
2. The approach leverages advanced image processing techniques and deep learning models to classify images and identify patterns associated with these diseases.

## Aim
1. The primary aim of this project is to develop a robust system for the automatic classification of histopathological images to assist in the diagnosis of autoimmune diseases.
2. By utilizing Convolutional Neural Networks (CNNs) and pre-trained models like VGG16 and ResNet50, the project aims to achieve high accuracy in image classification and pattern recognition.

## Problem Statements
1. **Automated Image Classification for Autoimmune Disease Diagnosis**:
   Develop a deep learning model to automatically classify histopathological images into three categories: Lupus, Arthritis, and Sclerosis. This model should achieve high accuracy and robustness to assist pathologists in diagnosing autoimmune diseases.

2. **Feature Extraction and Pattern Recognition in Histopathological Images**:
   Utilize pre-trained CNN models like VGG16 and ResNet50 to extract relevant features from histopathological images. Identify and visualize patterns that are indicative of specific autoimmune diseases to support clinical decision-making.

3. **Image Preprocessing and Enhancement for Noise Reduction**:
   Implement image preprocessing techniques, such as Gaussian blur, to reduce noise in histopathological images. Evaluate the impact of these preprocessing steps on the performance of deep learning models in classifying the images.

4. **Evaluation of Model Performance and Pattern Analysis**:
   Evaluate the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score. Generate a classification report and confusion matrix to analyze the model's effectiveness in diagnosing autoimmune diseases. Additionally, develop methods to visualize identified patterns in the images for further analysis.

5. **Development of a Data Augmentation Pipeline for Histopathological Images**:
   Create a data augmentation pipeline using techniques like rotation, width shift, height shift, shear, zoom, and horizontal flip to enhance the training dataset. Assess the impact of data augmentation on the model's generalization ability and classification accuracy.


## Technologies Used
- Python
- OpenCV
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Seaborn

## Data
The dataset consists of histopathological images categorized into three classes: Lupus, Arthritis, and Sclerosis. 
The images are stored in respective directories:
Medical\Disease Images
Medical\Arthritis
Medical\Lupus
Medical\Sclerosis

## Methodology
1. **Data Preprocessing**: 
   - Images are read from the directories and preprocessed by applying Gaussian blur for noise removal.
   - Images are resized to 224x224 pixels to match the input size required by the pre-trained models.

2. **Feature Extraction**:
   - Pre-trained VGG16 model (without top layers) is used to extract features from the images.

3. **Model Building**:
   - A Sequential model is built on top of the extracted features, comprising dense and dropout layers.
   - The model is trained on the extracted features with categorical crossentropy loss and Adam optimizer.

4. **Evaluation**:
   - The model's performance is evaluated using classification report and confusion matrix.
   - Visualization of identified patterns in the images is performed to analyze the results.

## Installation
1. Clone the repository to your local machine:
    ```sh
    git https://github.com/swaps361/Analysis-of-Autoimmune-diseases.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Analysis-of-Autoimmune-diseases
    ```
3. Open Visual Studio Code (VS Code) and select "File" > "Open Folder" to open the cloned directory.

4. Install any required extensions for Python development in VS Code.
   
5. Run each cell to perform the workflow.

## Contact
For any questions or feedback, feel free to reach out to [swapnildas742@gmail.com](mailto:swapnildas742@gmail.com).
