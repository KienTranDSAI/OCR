import cv2
import json
import numpy as np

with open("test1.json", 'r') as file:
    txt_inf = json.load(file)


cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/279573828566031.mp4")
vidFrame = []
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        # frame = cv2.resize(frame, (595,595))
        width, height = int(cap.get(3)), int(cap.get(4))
        
        vidFrame.append(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

for key in txt_inf.keys():
    box = txt_inf[key]['box']
    frame = txt_inf[key]['frame']
    if len(frame) < 4:
        continue
 
    pts = np.array(box, dtype = np.int32)


    for i in frame:
        vidFrame[i] = cv2.polylines(vidFrame[i], [pts], 
                      True, (0,0,255), 2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_draw_box.mp4', fourcc, 30.0, (width,height))

for i in range(len(vidFrame)):
    out.write(vidFrame[i])
out.release()