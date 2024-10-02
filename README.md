# Age and Gender Detection using Deep Learning

## Overview:
This project implements an **Age and Gender Detection** system using **OpenCV** and **deep learning models**. The system can detect faces in images or video streams, predict the gender (Male/Female), and estimate the age group of detected individuals. It uses pre-trained deep learning models for face detection and classification of age and gender.

The project is designed for real-time or batch processing of images and videos, providing a flexible solution for various applications such as retail analytics, security systems, and demographic data collection.

## Key Features:
- **Face Detection**: Utilizes a pre-trained face detection model to locate faces in an image or video frame.
- **Age Prediction**: Predicts the age group from a set of predefined categories (e.g., 0-2, 4-6, 8-12, etc.).
- **Gender Prediction**: Classifies gender as either Male or Female.
- **Image and Video Processing**: Can handle individual image files as well as video files for batch processing.
- **Pre-trained Models**: Uses the **Caffe** models for age and gender classification, ensuring high accuracy and fast predictions.

## Models and Architecture:
- **Face Detection**: Uses OpenCV’s DNN face detector based on a pre-trained model.
- **Age Detection**: A convolutional neural network (CNN) is used, trained on the Adience dataset, to classify the age group of a person.
- **Gender Detection**: Another CNN is used for gender classification, trained on the same dataset.
  
## Technologies and Libraries Used:
- **OpenCV**: For face detection, image processing, and video handling.
- **Deep Learning Models**: Pre-trained models for face detection, age, and gender classification.
- **Python**: Main programming language.
- **Google Colab**: For testing and execution in a cloud environment.
  
## Instructions to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/misbah4064/age_and_gender_detection.git
  
2. Download the pre-trained models:
   ```bash
   !gdown https://drive.google.com/uc?id=1_aDScOvBeBLCn_iv0oxSO8X1ySQpSbIS
   !unzip modelNweight.zip

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the script to detect age and gender on an image:
   ```bash
   python main.py --input image.jpg

5. For video processing, use:
   ```bash
   python main.py --input video.mp4

## Datasets:
- The models are trained on the **Adience dataset**, which contains face images labeled with age and gender information. The dataset is used to train the age and gender models, providing predictions in predefined categories.

## How it Works:
- **Face Detection**: The face detection model is used to identify faces in the input image or video frame.
- **Age and Gender Classification**: Once a face is detected, the system extracts the face region and passes it through age and gender classification networks to predict the corresponding labels.
- **Output**: The predictions are drawn on the image/video with bounding boxes around the faces and the predicted age and gender displayed.

## Demo:
- **Process an image**:
  ```bash
    python main.py --input image.jpg
- **Process a video**:
   ```bash
    python main.py --input video.mp4

## Output Example:
Predicted age and gender will be displayed on the image or video frames with bounding boxes around the detected faces.
