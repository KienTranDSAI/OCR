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



def stabilize_ocr(previous_res, res, threshold = 50):
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
    # horizontal = False
    # if abs(coor1[0][1] - coor2[0][1]) < 50:
    #     horizontal = True
    if cen_dist < 150:
        return True
    if math.sqrt((coor1[1][0] - coor2[0][0])**2 + (coor1[1][1] - coor2[0][1])**2) < 100:
        return True
    if math.sqrt((coor1[3][0] - coor2[3][0])**2 + (coor1[3][1] - coor2[3][1])**2) < 100:
        return True
    return False
def check_horizontal(coor1, coor2):
    if abs(coor1[0][1] - coor2[0][1]) < 10:
        return True
    return False
def check_left(coor1, coor2):
    if abs(coor1[0][0] - coor2[0][0]) > 5:
        return False
    return True
def check_right(coor1, coor2):
    if abs(coor1[1][0] - coor2[1][0]) > 5:
        return False
    return True
def merge_box(result):
    res = [line[0] for line in result]
    cop_res = res[:]
    my_dict = {}
    same = {}
    for i in range(len(res)):
        my_dict[i] = [i]
        same[i] = [i]
    for i in range(len(res)):
        for j in range(i+1,len(res)):
            isMerge= check2merge(res[i], res[j])
            if isMerge:
                my_dict[i] = my_dict[i] + [j]
                my_dict[j] = my_dict[j] + [i]

                top_left = [min(res[i][0][0], res[j][0][0]), min(res[i][0][1], res[j][0][1])]
                top_right = [max(res[i][1][0], res[j][1][0]), min(res[i][1][1], res[j][1][1])]
                bot_right = [max(res[i][2][0], res[j][2][0]), max(res[i][2][1], res[j][2][1])]
                bot_left = [min(res[i][3][0], res[j][3][0]), max(res[i][3][1], res[j][3][1])]
                new_coor = [top_left, top_right,bot_right,bot_left]
                # res[i] = new_coor
                # res[j] = new_coor
                for k in my_dict[i]:
                    res[k] = new_coor
                for k in my_dict[j]:
                    res[k] = new_coor
    store = [False]*len(res)

    for i in range(len(result)):
        result[i][0] = res[i]
        for j in range(i+1,len(result)):
            if res[i] == res[j] and not store[j]:
                same[i] = same[i] + [j]
                store[j] = True
    new_boxes = []
    new_texts = []
    align = []
    is_process = [False]*len(res)
    for key in same.keys():
        if len(same[key]) > 1:  
            is_left = True
            is_right = True
            text = result[same[key][0]][1][0]
            is_process[same[key][0]] = True
            for i in range(1, len(same[key])):
                if check_horizontal(cop_res[same[key][i-1]], cop_res[same[key][i]]):
                    text += " " + result[same[key][i]][1][0]
                else:
                    text += " /n " + result[same[key][i]][1][0]
                    if not check_left(cop_res[same[key][i]], res[same[key][0]]):
                        is_left = False
                    if not check_right(cop_res[same[key][i]], res[same[key][0]]):
                        is_right = False
                    
              

                is_process[same[key][i]] = True
            if is_left and is_right:
                align_ = "Not determined align"
                print("Not determined align")
            elif is_left:
                align_ = "Left align"
                print("Left")
            elif is_right:
                align_ = "Right align"
                print('right')
            else:
                align_ = "Center align"
                print("center")
            print(text)
            new_texts.append(text)
            new_boxes.append(res[same[key][0]])
            align.append(align_)
    for i in range(len(res)):
        if not is_process[i]:
            new_texts.append(result[i][1][0])
            new_boxes.append(result[i][0])
            align.append("None")
    # print(new_boxes)
    # print(new_texts)
    return result, new_boxes, new_texts, align, cop_res


cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/404759268832213.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

ocr = PaddleOCR(use_angle_cls=False, lang="en")
filter_low_score = True
previous_res = None
store_all_frame = {}
store_dict = {}
restore = {}

frame_threshold = fps//2

countt = 0


while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
        # frame = cv2.resize(frame, (595,595))
        width, height = int(cap.get(3)), int(cap.get(4))
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Video", frame)
        result = ocr.ocr(grayFrame, cls=True)

        if result == [None]:
            countt += 1
            continue
        for idx in range(len(result)):
            res = result[idx]

        result = result[0]
        result = remove_low_score(result, 0.9)


        result, new_boxes, new_texts, align, cop_res = merge_box(result)
        result = [[new_boxes[i], new_texts[i]] for i in range(len(new_boxes))]
        result = stabilize_ocr(previous_res, result)
        previous_res = result[:]

        log = [[new_boxes[i], new_texts[i],align[i]] for i in range(len(new_boxes))]
        boxes = [line[0] for line in result]
        txts = [line[1].strip() for line in result]
        # scores = [line[1][1] for line in result]
        store_all_frame[countt] = log
        


        for i in range(len(result)):
            
            if txts[i] in store_dict.keys():
                # print(f"This word is in dict {txts[i]}")
                if boxes[i] != store_dict[txts[i]]['box']:
                    if store_dict[txts[i]]['end'] - store_dict[txts[i]]['start'] > frame_threshold:
                        restore[txts[i]] = store_dict[txts[i]]
                        # print(f"Store {[txts[i]]}")
                    store_dict[txts[i]] = {'box': boxes[i],
                                               'start':  countt,
                                               'end': countt + 1,
                                               'align': align[i]}
                    
                else:
                    store_dict[txts[i]]['end'] = store_dict[txts[i]]['end'] + 1
                    # print(f"add {txts[i]}")
            else: 
                store_dict[txts[i]] = {'box': boxes[i],
                                        'start':  countt,
                                        'end': countt + 1,
                                        'align': align[i]}

        RGBimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # im_show = draw_ocr(RGBimg, boxes, txts, scores, font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = draw_ocr(RGBimg, boxes ,font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("A", im_show)
        countt += 1
        if countt == 400:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
print(store_dict)
with open("raw_ocr.json", 'w') as file:
    json.dump(store_all_frame, file,indent=0, cls=CustomIndentEncoder)
with open("raw_ocr.pickle", 'wb') as file:
    pickle.dump(store_all_frame, file)

ans = {}
for key in store_dict.keys():
    if store_dict[key]['end'] - store_dict[key]['start'] > frame_threshold:
        ans[key] = store_dict[key]
for key in restore.keys():
    ans[key] = restore[key]   
with open("processed_results.json", 'w') as file:
    json.dump(ans, file, indent=0, cls=CustomIndentEncoder)   
cap.release()
cv2.destroyAllWindows()