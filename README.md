
# 👤 Face Recognition System Using SVM

## 🚀 Project Overview

This project is a **Face Recgition System** built using **Machine Learning (SVM)** that can identify and recognize faces of my family members in real-time.

The system captures images using a camera, processes them with OpenCV, extracts facial features, and classifies them using Support Vector Machine(SVM) model.

---

## 🧠Technologies Used 
* Pyhton
* OpenCV
* Scikit-Learn (SVM)
* face_recognition (for encoding)
* Flask (Backend API)
* HTML & CSS (Frontend UI)

---

## ⚙️Features
* Real-time face detection and recognition
* Custom dataset (family members)
* SVM-based classification model
* Model saving and loading using Pickle

---
## 🔄 Workflow
1. Collect images of each person and store in dataset/
2. Encode faces using face_recognition
3. Train SVM model using encodings
4. Save model using pickle
5. Start Flask server for live recognition
6. Open browser to view real-time results
---

## How to Run 
1. Clone Repository
```
git clone https://github.com/your-username/face-recognition-svm.git
cd face-recognition-svm
```

2. Install Dependencies
```
pip install -r requirements.txt
```
**Note** : I build this project using **python 3.10**.

3. Create Dataset
Run Script :
```
python src/dataset_creation.py
```

4. Train Model
```
python src/model_creation.py
```

5. Run Flask App
```
python src/backend/app.py
```

6. Open in Browser
```
http://127.0.0.1:5000
```
**Note** : I didn't deploy in server.So, I use localhost.

## Author
Herambha Karthikeya Guptha


