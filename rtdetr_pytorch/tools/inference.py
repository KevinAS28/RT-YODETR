import os
import subprocess
import argparse
import json
import time

import torch
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor

def get_gpu_memory():
  return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"], capture_output=True).stdout.decode('utf-8').split('\n')[1].split(' ')[0])

named_labels = {
    0: 'person',
    1: 'bicycle'
}

def inference_images(model_sess, imgs_dir, output_dir, size=(640, 640), font_file='DejaVuSans.ttf', thrh=0.6, box_color='red', text_color='blue', class_labels=json.dumps(named_labels), show_index=True, show_percent=True):
    class_labels = json.loads(class_labels)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    font = None
    try:
        font = ImageFont.truetype(font_file, 16)
    except OSError:
        raise FileNotFoundError(f'Cannot find font {font_file}, try download from https://get.fontspace.co/download/family/1137/58186c2f7a464684be77bb251fdf4062/swansea-font.zip')

    total_time_start = time.time()

    totensor = ToTensor()
    real_imgs = []
    for img_name in os.listdir(imgs_dir):
        img_path = os.path.join(imgs_dir, img_name)
        im = Image.open(img_path).convert('RGB')
        im = im.resize(size)
        real_imgs.append((img_name, im))

    imgs_preprocessed = torch.stack([totensor(i[1]) for i in real_imgs])
    inference_time_start = time.time()

    labels, boxes, scores = model_sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': imgs_preprocessed.data.numpy(), "orig_target_sizes": torch.tensor([[640, 640]]*len(real_imgs)).data.numpy()}
    ) 

    inference_time_final = time.time()-inference_time_start

    output_img_paths = []

    for i in range(len(real_imgs)):
        im = real_imgs[i][1]
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr_str = '-'+str(round(scr[i]*100, 4))+'%' if show_percent else ''
        lab_str = lab[i]+'-' if show_index else ''
        # print(scr_str, lab_str, len(box))
        for b in box:
            draw.rectangle(list(b), outline=box_color,)
            draw.text((b[0], b[1]), text=f'{lab_str}{named_labels[lab[i]]}{scr_str}', fill=text_color, font=font)

        out_file_path = os.path.join(output_dir, real_imgs[i][0])
        im.save(out_file_path)
        output_img_paths.append(out_file_path)
        print(f'{i+1}/{len(real_imgs)}: {out_file_path}')

    total_time_final = time.time() - total_time_start

    print(f'total time: {total_time_final:.4f}s | inference time: {inference_time_final:.4f}s')
    print(f'AVG total time: {(total_time_final/len(real_imgs)):.4f}s | inference time: {(inference_time_final/len(real_imgs)):.4f}s')

    return total_time_final, inference_time_final, output_img_paths


def main(args):
    print(f'onnx device: {ort.get_device()}')
    print(f'loading the model {args.model} ...')
    sess = ort.InferenceSession(args.model)
    print('start inference')
    inference_images(sess, args.imgs_dir, args.output_dir, (args.size, args.size), args.font_file, args.threshold, args.box_color, args.text_color, args.classes_dict, args.show_index, args.show_percent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, )
    parser.add_argument('--imgs_dir', '-i', type=str, )
    parser.add_argument('--output_dir', '-o', type=str, )
    parser.add_argument('--size', '-s', type=int, default=640)
    parser.add_argument('--font_file', '-f', type=str, default='DejaVuSans.ttf')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, help='a float number from 0 to 1 (0.6, 0.99)')
    parser.add_argument('--box_color', '-b', type=str, default='red')
    parser.add_argument('--text_color', type=str, default='blue')
    parser.add_argument('--show_index', action='store_true', default=False)
    parser.add_argument('--show_percent', action='store_true', default=False)
    parser.add_argument('--classes_dict', '-c', type=str, default=json.dumps(named_labels))

    args = parser.parse_args()

    main(args)

    # python3 tools/inference.py --model=rtdetr_yolov9bb_ep27.onnx --imgs_dir=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/imgs --output_dir=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/output --font_file=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/font/Swansea-q3pd.ttf -t 0.8 --text_color=white  --show_percent