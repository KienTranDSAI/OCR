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

def remove_small_boxes(result, threshold = 300):
    ind2rem = []
    ans = []
    for i in range(len(result)):
        width = result[i][0][1][0] - result[i][0][0][0]
        height =  result[i][0][2][0] - result[i][0][0][0]
        area = width * height
        if area < threshold:
            ind2rem.append(i)
    for i in range(len(result)):
        if i not in ind2rem:
            ans.append(result[i])
    return ans

def stabilize_ocr(previous_res, res, threshold = 20):
    if previous_res != None:
        for i in range(len(previous_res)):
            for j in range(len(res)):
                if compare_box(res[j][0], previous_res[i][0]) <  threshold:
                    res[j] = previous_res[i]
    return res

def check2merge(coor1, coor2):
    cen1  = [(coor1[0][0] + coor1[2][0])/2, (coor1[0][1] + coor1[1][1])/2]
    cen2  = [(coor2[0][0] + coor2[2][0])/2, (coor2[0][1] + coor2[1][1])/2]

    cen_dist = math.sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)
    
    if cen_dist < 200:
        return True
    return False

def merge_box(res):
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if check2merge(res[i], res[j]):
                top_left = [min(res[i][0][0], res[j][0][0]), min(res[i][0][1], res[j][0][1])]
                top_right = [max(res[i][1][0], res[j][1][0]), min(res[i][1][1], res[j][1][1])]
                bot_right = [max(res[i][2][0], res[j][2][0]), max(res[i][2][1], res[j][2][1])]
                bot_left = [min(res[i][3][0], res[j][3][0]), max(res[i][3][1], res[j][3][1])]
                new_coor = [top_left, top_right,bot_right,bot_left]
                res[i] = new_coor
                res[j] = new_coor
    return res
def merge_box_test(result):
    ind2rem = []
    for i in range(len(result)):
        for j in range(i, len(result)):
            if (i not in ind2rem) and (result[i][0] == result[j][0]):
                result[i][1] = (result[i][1][0] + result[j][1][0], result[i][1][1])
                ind2rem.append(j)
    ans = []
    for i in range(len(result)):
        if i not in ind2rem:
            ans.append(result[i])
    return ans


cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/279573828566031.mp4")
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
            # for line in res:
                # print(line)
        # show result

        result = result[0]
        if filter_low_score:
            result = remove_low_score(result, 0.8)
        # result = remove_small_boxes(result)
        result = stabilize_ocr(previous_res, result)
        previous_res = result

        
        boxes = [line[0] for line in result]
        # txts = [line[1][0] for line in result]
        # result = merge_box_text(boxes,result)
        boxes = merge_box(boxes)

        result[0] = boxes
        result = merge_box_test(result)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        RGBimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        im_show = draw_ocr(RGBimg, boxes, txts, scores, font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.save('result1.jpg')
        im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("A", im_show)
        countt += 1
        temp = {}
        for i in range(len(boxes)):
            temp[txts[i]] = {"--box": boxes[i], "--score": scores[i]}
        store_dict[countt] = temp
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

with open("no_processing.json", 'w') as file:
    json.dump(store_dict, file, indent=0, cls=CustomIndentEncoder)
cap.release()
cv2.destroyAllWindows()