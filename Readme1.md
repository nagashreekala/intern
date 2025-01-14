## Project Report: Face Recognition Attendance System

### Introduction
In today’s digital age, automating attendance systems is crucial for improving efficiency and reducing human error. This project implements a real-time face recognition attendance system using a pre-trained deep learning model, OpenCV for image processing, and MySQL for database management. The system recognizes individuals, marks their attendance, and stores the data securely.

### Objective
The primary objective of this project is to develop a robust and automated attendance system that leverages facial recognition technology to:
- Accurately identify individuals in real-time.
- Record attendance seamlessly into a database.
- Eliminate the need for manual processes.

### System Overview
The face recognition attendance system comprises three main components:
1. **Face Recognition Model**: A pre-trained TensorFlow model (`face_recog.h5`) for identifying faces.
2. **Webcam Integration**: Captures real-time images for recognition.
3. **Database Management**: A MySQL database stores attendance records.

### Implementation

#### Model Loading
The system uses a TensorFlow model pre-trained on a dataset for face recognition. The model is loaded using the `load_model()` function, ensuring high accuracy in identifying individuals.

```python
# Load the model
model = load_model("face_recog.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
```

#### Camera Integration
The system integrates with a webcam to capture real-time video frames. The captured frames are resized, normalized, and passed to the model for prediction.

```python
camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to capture image. Check your webcam.")
        break

    # Resize and preprocess image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # Predict
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
```

#### Attendance Recording
Attendance is recorded in a MySQL database. When a face is recognized with a confidence score exceeding a predefined threshold, the system checks if attendance has already been marked. If not, it inserts the record.

```python
def record_attendance(name):
    try:
        c.execute("SELECT * FROM attendance1 WHERE name = %s", (name,))
        if not c.fetchone():
            c.execute("INSERT INTO attendance1 (name, status) VALUES (%s, %s)", (name, 'present'))
            conn.commit()
            print(f"Attendance marked for {name}")
        else:
            print(f"Attendance already marked for {name}")
    except Exception as e:
        print(f"Error inserting attendance: {e}")
```

### Database Design
The MySQL database has a table named `attendance` with the following schema:
- **name**: Stores the name of the recognized individual.
- **status**: Indicates the attendance status (e.g., "present").

### Results
The system successfully identifies individuals in real-time and records their attendance. Below are example output images from the system:
![image](https://github.com/user-attachments/assets/77dcba87-c94b-4405-8933-a702047c7888)

![Screenshot 2025-01-14 212131](https://github.com/user-attachments/assets/69a86ae0-9526-44f9-8ab4-e043dadb9375)


### Conclusion
This face recognition attendance system demonstrates the potential of combining deep learning, image processing, and database management for automating routine tasks. The implementation is scalable and can be extended to larger datasets and multi-camera setups.


