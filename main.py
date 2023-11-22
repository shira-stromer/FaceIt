import cv2
import os
import numpy as np
import random
import pandas as pd
from utilities import save_object_as_json_file, load_json
import torch
from facenet_pytorch import MTCNN, extract_face
import matplotlib.pyplot as plt
from PIL import Image
import os



def get_wedding_images(json_file_path=None, read_image=True):
    image_directory = '/Users/shirabroner/Downloads/all-wedding-photos'

    if json_file_path is None:
        file_names = os.listdir(image_directory)

        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        image_files = [file for file in file_names if any(file.endswith(ext) for ext in image_extensions)]
        random.shuffle(image_files)
    else:
        image_files = list(load_json(json_file_path).keys())
    
    for file in image_files:
        image_path = os.path.join(image_directory, file)
        image = None
        if read_image:
            image = cv2.imread(image_path)
        print(f'Loading {image_path}')
        yield image_path, image

def get_classifiers():
    classifiers = []
    classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"))
    classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
    classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"))
    return classifiers


def label_number_of_faces(photo_num):
    labels = {}
    counter = 0
    face_classifiers = get_classifiers()
    for image_path, img in get_wedding_images():
        counter +=1
        print(counter)
        print(image_path, img.shape)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = [face_classifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(250, 250)) for face_classifier in face_classifiers]
        faces = [f for f in faces if len(f) != 0]
        if len(faces) != 0:
            faces = np.concatenate(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)        
        
        cv2.imshow("image", img)
        cv2.waitKey(0)    
        cv2.destroyAllWindows()
          
        labels[image_path] = int(input("how many labelable faces you see?").strip())
        
        if counter >= photo_num:
            save_object_as_json_file("faces_num_labels.json", labels)
            return
        pass 

def save_detected_faces():
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=torch.device('cpu' if torch.backends.mps.is_available() else 'cpu'))
    for file_path, _ in get_wedding_images(None, False):
        # Load an image
        image = Image.open(file_path)

        # Detect faces
        boxes, _ = mtcnn.detect(image)
        i = 0
        file_name = os.path.basename(file_path).split(".")[0]
        i = 0
        if boxes is not None:
            for box in boxes:
                x,y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
                print(h * w, (h * w) > 10000)
                if h * w < 10000:
                    continue

                extract_face(image, box, save_path=os.path.join('faces', f'{file_name}_{i}.png'.format(i)))
                i += 1


def find_the_best_face_detection_algorithm():
    face_classifiers = []
    #face_classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"))
    face_classifiers.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
    labels = load_json('faces_num_labels_1.json')
    data = []
    for image_path, img in get_wedding_images('faces_num_labels_1.json'):
        print(image_path, img.shape)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = [face_classifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(250, 250)) for face_classifier in face_classifiers]
        faces = [f for f in faces if len(f) != 0]
        len_faces = 0
        if len(faces) != 0:
            faces = np.concatenate(faces)
            len_faces = len(faces)
            #for (x, y, w, h) in faces:
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                #cv2.imshow("cropped", crop_img)
        
        data.append({"ImagePath": image_path, "DetectedFaces": len_faces, "LabeledFaces": labels[image_path]})
    df = pd.DataFrame(data)#.to_csv('FaceDetection.csv')
    print("Success:", 1 - (df['LabeledFaces'] - df['DetectedFaces']).apply(lambda x: max(x, 0)).sum() / df['LabeledFaces'].sum())
        #cv2.imshow("image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

#find_the_best_face_detection_algorithm()
save_detected_faces()
print('hi')
    
