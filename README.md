# 🎯 Face Recognition Attendance System

A Python-based Face Recognition Attendance System that automates attendance management using computer vision and facial recognition technology. The system identifies registered individuals in real time through a webcam and records their attendance, eliminating the need for manual attendance processes.

## 📌 Project Overview

The Face Recognition Attendance System uses facial recognition to identify individuals and mark attendance automatically. It captures live video, detects faces, matches them with stored images, and records attendance with the person's name, date, and timestamp.

This project demonstrates the practical application of computer vision and machine learning in automating everyday tasks.

## ✨ Features

- Real-time face detection using a webcam
- Face recognition of registered users
- Automatic attendance marking
- Date and time stamping
- Attendance stored in a CSV file
- Prevents duplicate attendance entries during a session
- Simple and user-friendly implementation

## 🛠️ Tech Stack

- Python
- OpenCV
- face_recognition
- NumPy
- Pandas
- CSV
- VS Code / Jupyter Notebook

## 📂 Project Structure

```
Face-Recognition-Attendance-System/
│
├── Images/
│   ├── person1.jpg
│   ├── person2.jpg
│
├── Attendance.csv
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Face-Recognition-Attendance-System.git
```

### 2. Navigate to the project folder

```bash
cd Face-Recognition-Attendance-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the application using:

```bash
python main.py
```

The application will:

1. Load images of registered individuals.
2. Generate facial encodings.
3. Open the webcam.
4. Detect and recognize faces in real time.
5. Record attendance with the current date and time.

## 📊 How It Works

1. Store images of registered users.
2. Encode facial features using the `face_recognition` library.
3. Capture live video through the webcam.
4. Detect faces in each frame.
5. Compare detected faces with stored encodings.
6. Mark attendance when a match is found.
7. Save attendance records in a CSV file.

## 📸 Sample Attendance Record

| Name | Date | Time |
|------|------|------|
| John Doe | 06-07-2026 | 10:15 AM |
| Jane Smith | 06-07-2026 | 10:18 AM |

## 🚀 Future Improvements

- Database integration (MySQL or SQLite)
- Web-based dashboard
- Email attendance reports
- Face registration module
- Anti-spoofing (liveness detection)
- Cloud-based attendance storage
- Multi-camera support

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

## 📄 License

This project is intended for educational and learning purposes.

## 👨‍💻 Author

**Kanak Kirti Sharma**

B.Tech in Computer Science

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and Data Analytics.

---

⭐ If you found this project useful, consider giving it a star!
