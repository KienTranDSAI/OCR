# Save data before and after postprocessing in json file
from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
import matplotlib.pyplot as plt
import math
import json
import pickle

#Format json file
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

##Remove low score 
def remove_low_score(result, threshold):
    ind2rem = []                        #Store index of box to remove
    ans = []                            #Store chosen boxes
    for i in range(len(result)):
        if result[i][1][1] < threshold:
            ind2rem.append(i)
    for i in range(len(result)):
        if i not in ind2rem:
            ans.append(result[i])
    return ans
    
    #Compare distance of 2 boxes base on coorinate of corners
def compare_box(coor1, coor2):
    score = 0
    for i in range(4):
        for j in range(2):
            score += abs(coor1[i][j] - coor2[i][j])
    return score


    #Stablize box with small change coordinate
def stabilize_ocr(previous_res, res, threshold = 50):
    if previous_res != None:
        for i in range(len(previous_res)):
            for j in range(len(res)):
                if compare_box(res[j][0], previous_res[i][0]) <  threshold:
                    res[j] = previous_res[i]
    return res

    #Check condition to merge 
def check2merge(coor1, coor2):
    cen1  = [(coor1[0][0] + coor1[2][0])/2, (coor1[0][1] + coor1[1][1])/2]          #Center of box 1
    cen2  = [(coor2[0][0] + coor2[2][0])/2, (coor2[0][1] + coor2[1][1])/2]          #Center of box 2

    cen_dist = math.sqrt((cen1[0] - cen2[0])**2 + (cen1[1] - cen2[1])**2)           #Distance between 2 boxes

    if cen_dist < 150:            
        return True
    
    #Check some special case
    if math.sqrt((coor1[1][0] - coor2[0][0])**2 + (coor1[1][1] - coor2[0][1])**2) < 100:     #First top right corner and second top right corner
        return True
    if math.sqrt((coor1[3][0] - coor2[3][0])**2 + (coor1[3][1] - coor2[3][1])**2) < 100:     #First bottom right corner vs second bottm right corner
        return True     
    return False

#Check whether two boxes in the same line or not
def check_horizontal(coor1, coor2):
    if abs(coor1[0][1] - coor2[0][1]) < 10:
        return True
    return False
#Check left aligh
def check_left(coor1, coor2):
    if abs(coor1[0][0] - coor2[0][0]) > 5:
        return False
    return True

#Check right align
def check_right(coor1, coor2):
    if abs(coor1[1][0] - coor2[1][0]) > 5:
        return False
    return True

#Merge 2 near boxes
def merge_box(result):
    res = [line[0] for line in result] #Contain updated coordinated
    cop_res = res[:]   #Store old coordinate of boxes
    group_dict = {}    #Store index of boxes that are merged together
    same_group = {}
    for i in range(len(res)):
        group_dict[i] = [i]
        same_group[i] = [i]
    for i in range(len(res)):
        for j in range(i+1,len(res)):
            isMerge= check2merge(res[i], res[j])
            if isMerge:
                group_dict[i] = group_dict[i] + [j]
                group_dict[j] = group_dict[j] + [i]

                #Determine outter coordinate
                top_left = [min(res[i][0][0], res[j][0][0]), min(res[i][0][1], res[j][0][1])]
                top_right = [max(res[i][1][0], res[j][1][0]), min(res[i][1][1], res[j][1][1])]
                bot_right = [max(res[i][2][0], res[j][2][0]), max(res[i][2][1], res[j][2][1])]
                bot_left = [min(res[i][3][0], res[j][3][0]), max(res[i][3][1], res[j][3][1])]
                new_coor = [top_left, top_right,bot_right,bot_left]
                
                #Update coordinates of all boxes in same group
                for k in group_dict[i]:
                    res[k] = new_coor
                for k in group_dict[j]:
                    res[k] = new_coor

    store = [False]*len(res)  #To indicate that old boxes is included in any group or not

    for i in range(len(result)):
        result[i][0] = res[i]
        for j in range(i+1,len(result)):
            if res[i] == res[j] and not store[j]:
                same_group[i] = same_group[i] + [j]
                store[j] = True
    
    new_boxes = []
    new_texts = []
    align = []

    is_process = [False]*len(res) #To indicate a box is in final box list or not
    for key in same_group.keys():
        
        if len(same_group[key]) > 1:  #Case: group of box
            is_left = True
            is_right = True
            text = result[same_group[key][0]][1][0]
            is_process[same_group[key][0]] = True
            for i in range(1, len(same_group[key])):
                if check_horizontal(cop_res[same_group[key][i-1]], cop_res[same_group[key][i]]):
                    text += " " + result[same_group[key][i]][1][0]
                else:
                    text += " /n " + result[same_group[key][i]][1][0]
                    if not check_left(cop_res[same_group[key][i]], res[same_group[key][0]]):
                        is_left = False
                    if not check_right(cop_res[same_group[key][i]], res[same_group[key][0]]):
                        is_right = False
                    
              

                is_process[same_group[key][i]] = True
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
            new_boxes.append(res[same_group[key][0]])
            align.append(align_)
    
    for i in range(len(res)):
        if not is_process[i]:           #Case: single box - unmerged boxes
            new_texts.append(result[i][1][0])
            new_boxes.append(result[i][0])
            align.append("None")
    return result, new_boxes, new_texts, align, cop_res


cap = cv2.VideoCapture("/home/kientran/Code/Work/OCR/Video/404759268832213.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
ocr = PaddleOCR(use_angle_cls=False, lang="en")
previous_res = None
store_all_frame = {}
store_dict = {}
restore = {}
frame_threshold = fps//2

count_frame = 0


while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
        width, height = int(cap.get(3)), int(cap.get(4))
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        result = ocr.ocr(grayFrame, cls=True)

        if result == [None]:
            count_frame += 1
            continue

        result = result[0]
        #Remove low score
        result = remove_low_score(result, 0.9)
        #Merge boxe
        result, new_boxes, new_texts, align, cop_res = merge_box(result)
        result = [[new_boxes[i], new_texts[i]] for i in range(len(new_boxes))]
        #Stablize boxes
        result = stabilize_ocr(previous_res, result)
        previous_res = result[:]

        log = [[new_boxes[i], new_texts[i],align[i]] for i in range(len(new_boxes))]  #For store results
        
        boxes = [line[0] for line in result]
        txts = [line[1].strip() for line in result]
        # scores = [line[1][1] for line in result]
        store_all_frame[count_frame] = log
        


        for i in range(len(result)):
            
            if txts[i] in store_dict.keys():
                # print(f"This word is in dict {txts[i]}")
                if boxes[i] != store_dict[txts[i]]['box']:
                    if store_dict[txts[i]]['end'] - store_dict[txts[i]]['start'] > frame_threshold:
                        restore[txts[i]] = store_dict[txts[i]]
                        # print(f"Store {[txts[i]]}")
                    store_dict[txts[i]] = {'box': boxes[i],
                                               'start':  count_frame,
                                               'end': count_frame + 1,
                                               'align': align[i]}
                    
                else:
                    store_dict[txts[i]]['end'] = store_dict[txts[i]]['end'] + 1
                    # print(f"add {txts[i]}")
            else: 
                store_dict[txts[i]] = {'box': boxes[i],
                                        'start':  count_frame,
                                        'end': count_frame + 1,
                                        'align': align[i]}

        RGBimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        im_show = draw_ocr(RGBimg, boxes ,font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
        im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", im_show)
        
        count_frame += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
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