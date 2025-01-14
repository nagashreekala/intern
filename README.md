### Project Report: Face Recognition Using Teachable Machine and Python

#### 1. **Introduction**
This project leverages Google Teachable Machine's capabilities to create a face recognition system. The trained model is integrated with a Python application, enabling real-time face detection and classification using a webcam.
![image](https://github.com/user-attachments/assets/038d5dd8-40bd-4fc2-9411-9a56fbcc626c)



#### 2. **Objective**
The goal is to implement a face recognition system that uses a pre-trained model from Google Teachable Machine. The system should:
- Detect faces in real-time using a webcam.
- Classify detected faces based on pre-trained labels.
- Provide confidence scores for predictions.
- ![image](https://github.com/user-attachments/assets/c57b983a-bd10-4e67-8d3d-d768534783d5


#### 3. **System Overview**
The project uses the following technologies:
- **Google Teachable Machine**: For model training and exporting.
- **Python**: For application logic and integration.
- **Keras and TensorFlow**: For loading and using the pre-trained model.
- **OpenCV**: For real-time image capture and processing.
- ![image](https://github.com/user-attachments/assets/d72cca69-dbbd-4c14-88ba-147af717630d)


#### 4. **Requirements**
- Python 3.x
- TensorFlow and Keras libraries
- OpenCV for image processing
- Trained model file (`face_recog.h5`)
- Class labels file (`labels.txt`)
- A webcam for real-time face detection
- ![image](https://github.com/user-attachments/assets/fca04fd1-cd51-48e9-ab93-a804b8be59e7)


#### 5. **Code Implementation**
Key features of the provided Python script:
1. **Model Loading**: 
   - Loads the `face_recog.h5` model.
   - ![image](https://github.com/user-attachments/assets/b2cadc17-c4cf-41d8-a982-bd3480e25c46)

   - Reads class labels from `labels.txt`.
   - ![image](https://github.com/user-attachments/assets/efef4ae3-c201-4c6c-8d00-9b3f2cd280bb)

2. **Webcam Initialization**: 
   - Captures real-time video feed from a connected webcam.
3. **Image Preprocessing**:
   - Resizes frames to 224x224 pixels.
   - Normalizes pixel values.
   - ![image](https://github.com/user-attachments/assets/7513f818-5db6-4139-b9fb-f77ec1048883)

4. **Prediction**:
   - Makes predictions using the loaded model.
   - Displays the predicted class and confidence score.
5. **User Interaction**:
   - Displays live feed with predictions.
   - Allows exiting with the 'Esc' key.
   - ![image](https://github.com/user-attachments/assets/084bb496-a441-432b-8104-83732ebea2ec)


#### 6. **Results**
The system successfully:
- Detects faces in real-time.
- Classifies them into predefined categories with confidence scores.
- Demonstrates efficient real-time performance.
- ![Screenshot 2025-01-10 110738](https://github.com/user-attachments/assets/e07db1c7-35de-4680-8db0-4dcd3ec9b68d)


#### 7. **Challenges**
- Dependency on proper lighting for accurate detection.
- High confidence scores may require balanced training data.

#### 8. **Conclusion**
The project illustrates the practical application of Google Teachable Machine models in Python for real-time face recognition. Future enhancements could include:
- Improved UI for better user interaction.
- Integration with additional datasets for broader classification capabilities.

#### 9. **References**
- Google Teachable Machine documentation
- TensorFlow and OpenCV libraries
![image](https://github.com/user-attachments/assets/ab3d9ed1-c756-4a7b-af77-b421834177c4)
