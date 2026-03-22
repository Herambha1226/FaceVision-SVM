from sklearn.svm import SVC
import os
import face_recognition 
import pickle

class Face_Recognation():
    def face_encoding(self):
        x = []
        y = []

        dataset_path = "dataset"

        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path,person)

            for image in os.listdir(person_path):
                image_path = os.path.join(person_path,image)

                image = face_recognition.load_image_file(image_path)
                if image is None:
                    print(f"There is No Image in {person} folder !")
                    continue
                encoding = face_recognition.face_encodings(image)
                

                if len(encoding) > 0:
                    x.append(encoding[0])
                    y.append(os.path.basename(person_path))
            print(f"Successfully encoding {person} !")
        print("Encoding of Images Completed Successfully !")

        return x,y
    
    def model_creation(self):

        x,y = self.face_encoding()

        model = SVC(kernel='linear')
        model.fit(x,y)

        with open("model/model.pkl",'wb') as f:
            pickle.dump(model,f)

        print("Model Created Successfully !")


if __name__ == "__main__":
    obj = Face_Recognation()
    obj.model_creation()


