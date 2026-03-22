from flask import Flask,render_template,Response
import os
from dotenv import load_dotenv
import cv2 
import pickle as pkl
import face_recognition

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

cap = cv2.VideoCapture(0)

with open("model/model.pkl",'rb') as f:
    model = pkl.load(f)

def generate_frame():
    while True:
        success,frame = cap.read()

        if not success:
            print("There is Problem in getting Frame")
        else:

            small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            face_location = face_recognition.face_locations(rgb_frame,model='hog')
            face_encoding = face_recognition.face_encodings(rgb_frame,face_location)

            for (top,right,bottom,left),encoding in zip(face_location,face_encoding):
                if len(encoding) > 0:
                    prediction = model.predict([encoding])
                    #print(f"Prediction Person : {prediction}")

                    cv2.rectangle(frame,(left,top),(right,bottom),(245, 222, 179),6)

                    cv2.putText(frame,str(prediction[0]),(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(145, 203,242),4)


            _, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("main.html")
            

@app.route('/video')
def video():
    return Response(generate_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)