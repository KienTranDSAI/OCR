# Save data before and after postprocessing in json file
from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
import matplotlib.pyplot as plt
import math
import json
import pickle

class CustomIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        # Override the default encoder with no indentation
        super().__init__(*args, **kwargs)
    
    def encode(self, obj):
        # If the object is a dictionary, add a newline before each key
        if isinstance(obj, dict):
            items = [f'\n"{key}": {self.encode(value)}' for key, value in obj.items()]
            return '{' + ','.join(items) + '\n}'
        return super().encode(obj)
    
def remove_low_score(result, threshold):
    # for line in result:
    #     if (line[1][1] < threshold):
    #         result.remove(line)
    # return result
    ind2rem = []
    ans = []
    for i in range(len(result)):
        if result[i][1][1] < threshold:
            ind2rem.append(i)
    for i in range(len(result)):
        if i not in ind2rem:
            ans.append(result[i])
    return ans
def compare_box(coor1, coor2):
    score = 0
    for i in range(4):
        for j in range(2):
            score += abs(coor1[i][j] - coor2[i][j])
    return score



def stabilize_ocr(previous_res, res, threshold = 40):
    if previous_res != None:
        for i in range(len(previous_res)):
            for j in range(len(res)):
                if compare_box(res[j][0], previous_res[i][0]) <  threshold:
                    res[j] = previous_res[i]
    return res


cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/1496934080857005.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

ocr = PaddleOCR(use_angle_cls=False, lang="en")
filter_low_score = True
previous_res = None
store_dict = {}
countt = 0


while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.resize(frame, (595,595))
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Video", frame)
        result = ocr.ocr(grayFrame, cls=True)

        if result == [None]:
            continue
        for idx in range(len(result)):
            res = result[idx]

        result = result[0]
        store_dict[countt] = result
        boxes = [line[0] for line in result]
        txts = [line[1][0].strip() for line in result]
        scores = [line[1][1] for line in result]
       
        
        RGBimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # im_show = draw_ocr(RGBimg, boxes, txts, scores, font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = draw_ocr(RGBimg, boxes,font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("A", im_show)
        countt += 1
        if countt == 100:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

with open("raw_ocr.json", 'w') as file:
    json.dump(store_dict, file,indent=0, cls=CustomIndentEncoder)
with open("raw_ocr.pickle", 'wb') as file:
    pickle.dump(store_dict, file)
cap.release()
cv2.destroyAllWindows()