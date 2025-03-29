import cv2
import numpy as np
import torch
import os
from model_CNN import TinyVGG_01
from torchvision import transforms
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

HIDDEN_UNITS = 350

cap = cv2.VideoCapture(0)

target_dir = "models"
model_name_file = "tinyvgg_model_01.pth"
path_model = os.path.join(target_dir, model_name_file)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
transform = transforms.Compose([
    transforms.Resize(size=(48, 48)),
    transforms.ConvertImageDtype(dtype=torch.float32)
])

model = TinyVGG_01(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)
model.load_state_dict(torch.load(path_model, weights_only=True))
model.eval()

head_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while(True):

    ret, frame = cap.read() # frame by frame

    colored  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    head = head_cascade.detectMultiScale(colored, scaleFactor=1.05, minNeighbors=5, minSize=[40,40])

    for(x,y,w,h) in head:
        img_rectangle = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        head_frame = frame[y:y+h,x:x+w]
        head_frame_tensor = torch.from_numpy(head_frame)
        head_frame_tensor = head_frame_tensor.permute(2,1,0)
        head_frame_tensor_transormed = transform(head_frame_tensor)
        head_frame_tensor_transormed = head_frame_tensor_transormed.unsqueeze(dim=0).to(device)
        model_pred = model(head_frame_tensor_transormed)
        probabilities = F.softmax(model_pred, dim=1)
        values, _ = probabilities.max(dim=1)
        probabilitie_of_emoton = int(values.item()*100)
        emotion = model_pred.argmax(dim=1).item()
        cv2.putText(img_rectangle, str(class_names[emotion]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(img_rectangle, str(probabilitie_of_emoton), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
