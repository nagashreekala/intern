Report on a task
---

**Project Report: Face Recognition-Based Attendance System**

**Abstract**
This project implements an attendance management system using face recognition technology. The system identifies individuals using a pre-trained neural network model and marks their attendance in a MySQL database. It enhances traditional attendance methods by providing an automated, efficient, and secure solution.

---

#### **Introduction**
The manual process of recording attendance can be time-consuming and prone to errors. This project leverages artificial intelligence and computer vision to automate attendance recording. The system uses a face recognition model trained via Google Teachable Machine and integrates it with a MySQL database to maintain accurate attendance records.
![image](https://github.com/user-attachments/assets/f95f1507-0749-4c40-ba9c-4a31e08216d1)

---

#### **System Design**

##### **1. Architecture**
The system architecture comprises:
1. **Face Recognition Model**: A deep learning model trained using Google Teachable Machine to recognize faces.
2. **Database**: A MySQL database for storing attendance records.
3. **Webcam Interface**: Captures live video feed for face recognition.

##### **2. Key Modules**
- **Face Detection and Recognition**: Detects faces in real-time using a webcam and recognizes them using the pre-trained model.
- **Attendance Recording**: Marks attendance in the database if the recognition confidence exceeds a specified threshold.
- **User Feedback**: Displays recognition results and attendance status on a live webcam feed.

---

#### **Implementation**

##### **1. Technologies Used**
- **Programming Language**: Python
- **Libraries**: 
  - `cv2`: For video capture and image processing.
  - `numpy`: For numerical operations and data transformation.
  - `mysql.connector`: For database interaction.
  - `tensorflow.keras`: For loading and using the face recognition model.
- **Database**: MySQL

##### **2. Workflow**
1. **Model Loading**: Load the pre-trained model and label mappings.
2. **Image Preprocessing**: Capture and resize images from the webcam for model compatibility.
3. **Prediction**: Identify the individual using the model's predictions.
4. **Attendance Recording**: Update the database with the recognized individual's attendance status, ensuring no duplicate entries for the day.
5. **Live Feedback**: Display recognition results and confidence on the webcam feed.

##### **3. Code Highlights**
- **Real-time Face Recognition**:
  ```python
  image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
  image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
  prediction = model.predict(image_array, verbose=0)
  ```
- **Attendance Recording**:
  ```python
  def record_attendance(name):
      c.execute("INSERT INTO attendance (name, status) VALUES (%s, %s)", (name, 'present'))
      conn.commit()
  ```

---

#### **Results**
The system successfully:
1. Recognizes individuals with high confidence.
2. Records attendance in the database.
3. Provides a user-friendly interface for real-time feedback.
![image](https://github.com/user-attachments/assets/734eb784-fe2d-4e19-acce-3caabcb595fb)

---

#### **Conclusion**
The face recognition-based attendance system simplifies the process of attendance management by automating identification and record-keeping. Future enhancements could include:
- Integrating cloud storage for remote access.
- Adding support for multiple cameras.
- Implementing advanced security measures to prevent spoofing.

---

#### **References**
- OpenCV Documentation
- TensorFlow Documentation
- MySQL Documentation

Would you like me to include diagrams, expand on specific sections, or format it further?
