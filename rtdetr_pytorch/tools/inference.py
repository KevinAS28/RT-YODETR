import os 
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
import onnxruntime as ort
from PIL import Image, ImageDraw

import torch
import torch.nn as nn 
from torchvision.transforms import ToTensor


def perform_inference(model_path, images_path , output_dir, output_names=None):
    preprocessed_imgs = []
    # for i, img_name in enumerate(images_path):
    img_path = images_path#os.path.join(images_path, img_name)
    im = Image.open(img_path).convert('RGB')
    im = im.resize((640, 640))
    im_data = ToTensor()(im)[None]

    sess = ort.InferenceSession(model_path)
    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=output_names,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    )

    labels, boxes, scores = output

    draw = ImageDraw.Draw(im)
    thrh = 0.6
    detected_labs = []
    for i in range(im_data.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        print(i, sum(scr > thrh))

        for b in box:
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )
            detected_labs.append(str(lab[i]))
    im.save(os.path.join(output_dir, img_path))

# python tools/export_onnx.py  -c /home/kevin/Custom-RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_yolov9bb_L_cocotrimmed.yml --check -f rtdetr_yolov9bb_24.onnx -r /home/kevin/Custom-RT-DETR/rtdetr_pytorch/checkpoint0024.pth
print(perform_inference('test.onnx', 'bicycle.jpeg'))

