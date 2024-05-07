from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

ocr = PaddleOCR(use_angle_cls=False, lang="en") # need to run only once to download and load model into memory
img_path = 'images.jpeg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
result = ocr.ocr(img, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# show result
result = result[0]
# image = Image.open(img_path).convert('RGB')
# print(type(image))
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
im_show = draw_ocr(img, boxes, txts, scores, font_path='/home/kientran/Code/Work/OCR/latin.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result1.jpg')
