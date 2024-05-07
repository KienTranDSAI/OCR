# Save data before and after postprocessing in json file
from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
import matplotlib.pyplot as plt
import math
import json

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


cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/279573828566031.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

ocr = PaddleOCR(use_angle_cls=False, lang="en")
filter_low_score = True
previous_res = None
store_dict = {}
countt = 0
frame_threshold = fps//2
restore = {}

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (595,595))

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
        if filter_low_score:
            result = remove_low_score(result, 0.8)
        result = stabilize_ocr(previous_res, result)
        previous_res = result

        

        # print(result)
        boxes = [line[0] for line in result]
        txts = [line[1][0].strip() for line in result]
        scores = [line[1][1] for line in result]
        # try:
        #     print("--------------------")
        #     print(boxes(txts.index("RN")))
        #     print("---------------------")
        # except:
            # pass
        # print(store_dict)
        for i in range(len(result)):
            
            if txts[i] in store_dict.keys():
                # print(f"This word is in dict {txts[i]}")
                if boxes[i] != store_dict[txts[i]]['box']:
                    if store_dict[txts[i]]['end'] - store_dict[txts[i]]['start'] > frame_threshold:
                        restore[txts[i]] = store_dict[txts[i]]
                        # print(f"Store {[txts[i]]}")
                    store_dict[txts[i]] = {'box': boxes[i],
                                               'start':  countt,
                                               'end': countt + 1}
                    
                else:
                    store_dict[txts[i]]['end'] = store_dict[txts[i]]['end'] + 1
                    # print(f"add {txts[i]}")
            else: 
                store_dict[txts[i]] = {'box': boxes[i],
                                        'start':  countt,
                                        'end': countt + 1}
        
        RGBimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # im_show = draw_ocr(RGBimg, boxes, txts, scores, font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = draw_ocr(RGBimg, boxes,font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("A", im_show)
        out.write(im_show)
        countt += 1
        # temp = {}
        # for i in range(len(boxes)):
        #     temp[txts[i]] = {"--box": boxes[i], "--score": scores[i]}
        # store_dict[countt] = temp
        if countt == 100:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
ans = {}
for key in store_dict.keys():
    if store_dict[key]['end'] - store_dict[key]['start'] > frame_threshold:
        ans[key] = store_dict[key]
for key in restore.keys():
    ans[key] = restore[key]   
with open("result.json", 'w') as file:
    json.dump(ans, file, indent=0, cls=CustomIndentEncoder)
print(restore)
cap.release()
out.release()
cv2.destroyAllWindows()