import json
import cv2
import numpy as np
with open('processed_results.json', 'r') as file:
    myDict = json.load(file)

cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/404759268832213.mp4")
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

for key in myDict.keys():
    box = myDict[key]['box']
    pts = np.array(box, dtype =np.int32)
    startF = myDict[key]['start']
    endF = myDict[key]['end']
    for i in range(startF, endF+1):
        vidFrame[i] = cv2.polylines(vidFrame[i], [pts], 
                      True, (0,0,255), 2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_draw_box.mp4', fourcc, 30.0, (width,height))

for i in range(len(vidFrame)):
    out.write(vidFrame[i])
out.release()
