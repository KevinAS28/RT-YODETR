import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import subprocess
import argparse
import json
import time

import torch
from torch import nn
import onnxruntime as ort
from torchvision.transforms import v2 as trfmv2
from torchvision.io import read_image
import cv2
import shutil

from src.core import YAMLConfig

def get_gpu_memory():
  return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"], capture_output=True).stdout.decode('utf-8').split('\n')[1].split(' ')[0])

def get_ort_session(model_path):
    print('ONNX device:', ort.get_device())
    providers = ['CUDAExecutionProvider']
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    return ort_session

def get_torch_model(cfg_path, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Torch device:', device, torch.cuda.get_device_name() if device=='cuda' else 'CPU')

    cfg = YAMLConfig(cfg_path, resume=model_path)
    checkpoint = torch.load(model_path, map_location=device) 
    if 'ema' in checkpoint:
        print('Using EMA from state')
        state = checkpoint['ema']['module']
    else:
        print('not using EMA')
        state = checkpoint['model']        
    
    cfg.model.load_state_dict(state)

    class RTDETRModelDeploy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)    
            
    model = RTDETRModelDeploy().to(device)        

    return model, device

def inference_imgs_torch(model, input_data, size, device):
    orig_target_sizes = torch.tensor([[size, size]]).to(device)
    labels, boxes, scores = model(input_data.to(device), orig_target_sizes)
    return labels, boxes, scores

def inference_imgs_onnx(ort_session, input_data, size):
    orig_target_sizes = torch.tensor([[size, size]])
    labels, boxes, scores = ort_session.run(None, {'images': input_data.data.numpy(), 'orig_target_sizes': orig_target_sizes.data.numpy()})
    return labels, boxes, scores

def isjson(json_content):
    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        return False

def inference_images(inference_engine, imgs_dir, output_dir, class_labels, size=640, thrh=0.6, show_index=True, show_percent=True):

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    

    total_time_start = time.time()

    preprocess_transformations = trfmv2.Compose([
        trfmv2.ToImageTensor(),
        trfmv2.ConvertImageDtype(),    
        trfmv2.Resize(size=(size, size), antialias=True),
    ])

    real_imgs = []
    for img_name in os.listdir(imgs_dir):
        img_path = os.path.join(imgs_dir, img_name)
        im = read_image(img_path)
        real_imgs.append((img_name, im))

    preprocessed_frames = [preprocess_transformations(i[1]) for i in real_imgs]
    stacked_frames = torch.stack(preprocessed_frames)
    inference_time_start = time.time()

    labels, boxes, scores = inference_engine(stacked_frames, size)

    inference_time_final = time.time()-inference_time_start

    output_img_paths = []

    for i in range(len(real_imgs)):
        im = cv2.resize(cv2.cvtColor(real_imgs[i][1].permute(1,2,0).data.numpy(), cv2.COLOR_BGR2RGB), (640, 640))
        scr = scores[i]
        lab = labels[i]
        box = boxes[i]
        # print(scr_str, lab_str, len(box))
        for s, l, b in zip(scr, lab, box):
            if s<thrh:
                continue
            
            s, l = float(s), int(l)
            b = [int(j) for j in b]            

            lab_str = l+'-' if show_index else ''
            scr_str = '-'+str(round(s*100, 4))+'%' if show_percent else ''
            b = [int(i) for i in b]
            postprocessed_frame = cv2.rectangle(im , tuple(b[:2]), tuple(b[2:4]), color=(0, 0, 255), thickness=2)  # Red rectangle
            postprocessed_frame = cv2.putText(postprocessed_frame, f"{lab_str}{class_labels[l]}{scr_str}", tuple(b[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text                    

        out_file_path = os.path.join(output_dir, real_imgs[i][0])
        cv2.imwrite(out_file_path, postprocessed_frame)
        output_img_paths.append(out_file_path)
        print(f'{i+1}/{len(real_imgs)}: {out_file_path}')

    total_time_final = time.time() - total_time_start

    print(f'total time: {total_time_final:.4f}s | inference time: {inference_time_final:.4f}s')
    print(f'AVG total time: {(total_time_final/len(real_imgs)):.4f}s | inference time: {(inference_time_final/len(real_imgs)):.4f}s')

    return total_time_final, inference_time_final, output_img_paths




def main(args):
    if args.engine=='onnx':
        ort_session = get_ort_session(args.model)
        inference_engine = lambda stacked_imgs, size: inference_imgs_onnx(ort_session, stacked_imgs, size)
    elif args.engine=='torch':
        model, device = get_torch_model(args.model_conf, args.model)
        inference_engine = lambda stacked_imgs, size: inference_imgs_torch(model, stacked_imgs, size, device)
    else:
        raise ValueError('Invalid engine value')

    if os.path.isfile(args.classes_labels):
        with open(args.classes_labels, 'r') as cd_file:
            classes_labels = json.loads(cd_file.read())
    elif isjson(args.classes_labels):
        classes_labels = json.loads(args.classes_labels)
    else:
        raise ValueError('Invalid JSON classes labels')
    classes_labels = {v:k for k, v in classes_labels.items()}
    print(classes_labels)

    print('start inference')
    inference_images(inference_engine, args.imgs_dir, args.output_dir, classes_labels, args.size, args.threshold, args.show_index, args.show_percent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, )
    parser.add_argument('--engine', '-e', type=str, default='torch', help='torch or onnx')
    parser.add_argument('--imgs-dir', '-i', type=str, )
    parser.add_argument('--output-dir', '-o', type=str, )
    parser.add_argument('--model-conf', '-mc', type=str, default='rtdetr_cyolov9ebb_L_cocotrimmed.yml')
    parser.add_argument('--size', '-s', type=int, default=640)
    parser.add_argument('--threshold', '-t', type=float, default=0.7)
    parser.add_argument('--show-index', action='store_true', default=False)
    parser.add_argument('--show-percent', action='store_true', default=False)
    parser.add_argument('--classes-labels', '-c', type=str, default='inference_class_labels', help='name_label: index_label | by json path or the json string')

    args = parser.parse_args()

    main(args)


# python3 tools/inference.py --model=rtdetr_yolov9ebb_ep27.pth --imgs-dir=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/imgs --output-dir=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/output --engine=torch --model-conf=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_cyolov9ebb_L_cocotrimmed.yml --classes-labels=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/tools/inference_class_labels.json -t 0.8  --show-percent
