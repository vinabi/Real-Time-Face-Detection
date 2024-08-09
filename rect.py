import cv2
import face_recognition
import os
known_names=[]
known_enc=[]
path=r'C:\Users\MCR-Instructor PC\Desktop\images'
images=os.listdir(path)
for image in images:
    img=cv2.imread(os.path.join(path,image))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    known_names.append(image)
    known_enc.append(encodings[0])

cap=cv2.VideoCapture(0)

while True:
  ret,img=cap.read()
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  boxes=face_recognition.face_locations(rgb)
  if len(boxes)>0:
    enc=face_recognition.face_encodings(rgb,boxes)[0]
    matches = face_recognition.compare_faces(known_enc,enc,tolerance=0.55)
    print(matches)
    name='unknown'
    for i in range(0,len(matches)):
        if matches[i]:
            name=known_names[i]
            img = cv2.putText(img.copy(), name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    cv2.imshow('',img)
    cv2.waitKey(1)
