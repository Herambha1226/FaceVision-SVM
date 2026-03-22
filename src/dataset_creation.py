import face_recognition as FR 
import os 
import pandas as pd

x = []
y = []

dataset_path = "dataset"

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        print(f"Processing: {image_path}")

        image = FR.load_image_file(image_path)

        encoding = FR.face_encodings(image)

        if len(encoding) > 0:
            x.append(encoding[0])
            y.append(person)
        else:
            print(f"  ⚠️ No face detected in: {image_path}")  # ✅ add this

print("Encoding Successful!")

data = {
    "encoding": x,
    "person": y
}

df = pd.DataFrame(data)
print(df)
