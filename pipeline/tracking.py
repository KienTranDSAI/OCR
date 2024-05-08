
from ocr import get_ocr
import json
from difflib import SequenceMatcher
import re
def similar_difflib(a, b):
    return SequenceMatcher(None, a, b).ratio()

# def normaline_text(text):
#     return text.lower()
def normalize_text(l_string):
    # convert to lower case
    lower_string = l_string.lower()
    # remove all punctuation except words and space
    pattern = "[^a-zA-Z\d$/@.' ]"
    no_punc_string = re.sub(pattern,'', lower_string) 
    # remove white spaces
    no_wspace_string = no_punc_string.strip()
    # convert string to list of words
    lst_string = [no_wspace_string][0].split()
    # print(lst_string)
    
    # remove stopwords
    no_stpwords_string= "".join(lst_string)
    # for i in lst_string:
    #     if not i in stop_words:
    #         no_stpwords_string += [i]
    # # removing last space
    # no_stpwords_string = no_stpwords_string[:-1]
    return no_stpwords_string

def calculate_IoU_score(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[2][0], boxB[2][0])
    yB = min(boxA[2][1], boxB[2][1])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2][0] - boxA[0][0]) * (boxA[2][1] - boxA[0][1])
    boxBArea = (boxB[2][0] - boxB[0][0]) * (boxB[2][1] - boxB[0][1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (interArea / float(boxAArea + boxBArea - interArea)) if (float(boxAArea + boxBArea - interArea) !=0) else (interArea /0.0001)

    # Return the intersection over union value
    return iou

def check_diff_txt(txt1, txt2, threshold = 0.9):
    return similar_difflib(txt1, txt2) >= threshold

def check_IoU_score(coor1, coor2, threshold = 0.7):
    return calculate_IoU_score(coor1, coor2) >= threshold

def tracking(result: list):                        #Result: [frame[box, (text, score)]]
    
    text_info_dict = {}

    last_key = -1
    with open('log.txt', 'w') as file:
        for frame_count in range(len(result)):
            str_frame_count = str(frame_count)
            if result.get(str_frame_count, []) != []:
                boxes = [line[0] for line in result[str_frame_count]]
                txts = [line[1][0] for line in result[str_frame_count]]
                normalized_txts = [normalize_text(txt) for txt in txts]

                # temp_obj = []

                for box_id in range(len(result[str_frame_count])):
                    box_coor = boxes[box_id]
                    box_normalized_txt = normalized_txts[box_id]
                    isInDict = False
                    for key in text_info_dict.keys():
                        diff = similar_difflib(text_info_dict[key]['text_normalized'], box_normalized_txt)
                        iou_score = calculate_IoU_score(text_info_dict[key]['box'], box_coor)
                        file.write(f"{text_info_dict[key]['text_origin']} vs {txts[box_id]}: {diff}  {iou_score} \n")
                        if check_diff_txt(text_info_dict[key]['text_normalized'], box_normalized_txt) and check_IoU_score(text_info_dict[key]['box'], box_coor):
                            isInDict = True
                            text_info_dict[key]['frame'] = text_info_dict[key]['frame'] + [frame_count]
                            break
                            
                        
                    if not isInDict:
                        temp_obj = {
                                    "text_origin": txts[box_id],
                                    "text_normalized": box_normalized_txt,
                                    "box": box_coor,
                                    'frame': [frame_count]
                                }
                        text_info_dict[last_key+1] = temp_obj
                        last_key += 1

    return text_info_dict
if __name__ == "__main__":
    # with open("/home/kientran/Code/Work/OCR/pipeline/raw_results/279573828566031.json", 'r') as file:
    #     myDict = json.load(file)
    # test = tracking(myDict)
    # with open("test1.json", 'w') as file:
    #     json.dump(test, file)
    # print(test)
    a = [[
                215.0,
                396.0
            ],
            [
                254.0,
                357.0
            ],
            [
                275.0,
                377.0
            ],
            [
                235.0,
                416.0
            ]]
    b = [[
                215.0,
                396.0
            ],
            [
                254.0,
                357.0
            ],
            [
                275.0,
                377.0
            ],
            [
                235.0,
                416.0
            ]]
    print(calculate_IoU_score(a,b))

                


