import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import time
import argparse
import json

import cv2
import torch
import torch.nn as nn 
from torchvision.transforms import v2 as trfmv2
import onnxruntime as ort 

from src.core import YAMLConfig

def get_ort_session(model_path):
    # ort.set_default_logger_severity(1)
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

def stream_video(video_path, inference_engine, size, classes_labels, encoder='XVID', thrh=0.65, show_stream=False, out_video_path='', draw_obj_name=True, draw_obj_conf=True):
    cap = cv2.VideoCapture(video_path)
    save_video_output = len(out_video_path)>0

    if save_video_output:
        if os.path.isfile(save_video_output):
            os.remove(save_video_output)
        fourcc_code = cv2.VideoWriter_fourcc(*encoder)
        video_writer = cv2.VideoWriter(out_video_path, fourcc_code, 25.0, (size, size))  # Adjust FPS if needed
    
    if not cap.isOpened():
        print("Error opening video!")
        exit()

    start_time = time.time()
    total_inference_time = 0
    frame_count = 1
    eplased_time = 1
    fps = 0
    detected_class_frame = {
        name:0 for name in classes_labels.values()
    }

    preprocess_transformations = trfmv2.Compose([
        trfmv2.ToImageTensor(),
        trfmv2.ConvertImageDtype(),    
        trfmv2.Resize(size=(size, size), antialias=True),
    ])

    print('Stream started')
    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                print("Video stream ended or error!")
                break
            if (type(frame)==bool) or (frame is None):
                print('bad frame 0')
                continue
            
            inference_start_time = time.time()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preprocessed_frame = preprocess_transformations(frame)
            stacked_frames = torch.stack([preprocessed_frame]) 

            labels, boxes, scores = inference_engine(stacked_frames, size)
            inference_time = time.time()-inference_start_time
            total_inference_time += inference_time
            img_index = 0

            scr = scores[img_index]
            lab = labels[img_index]
            boxes = boxes[img_index]

            postprocessed_frame = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
            frame_count += 1
            fps = frame_count / eplased_time

            if (show_stream or save_video_output)  and len(lab)>0:
                for s, l, b in zip(scr, lab, boxes):
                    if s<=thrh:
                        continue

                    s, l = float(s), int(l)
                    b = [int(j) for j in b]
                    # print('s l b', s, l, b)

                    lab_str = str(classes_labels[l]) if draw_obj_name else ''
                    scr_str = ('-' if draw_obj_name else '' + str(round(s*100, 1))+'%') if draw_obj_conf else ''
                    
                    postprocessed_frame = cv2.rectangle(postprocessed_frame, tuple(b[:2]), tuple(b[2:4]), color=(0, 0, 255), thickness=2)  # Red rectangle
                    postprocessed_frame = cv2.putText(postprocessed_frame, f"{lab_str}{scr_str}", tuple(b[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text        

                    detected_class_frame[classes_labels[l]] += 1                        

            if (show_stream or save_video_output):
                postprocessed_frame = cv2.putText(postprocessed_frame, f"FPS: {fps:.2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # White text           

            if show_stream:
                cv2.imshow("Video with Object Detection", postprocessed_frame)    
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            else:
                if frame_count%100==0:
                    print(f"FPS: {fps:.2f}")

            if save_video_output:
                video_writer.write(postprocessed_frame)


            eplased_time = time.time()-start_time

        except KeyboardInterrupt:
            print('Keyboard interrupt')
            break

    if save_video_output:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    avg_inference_time = total_inference_time/frame_count

    return frame_count, eplased_time, fps, avg_inference_time, detected_class_frame

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

    frame_count, eplased_time, fps, avg_inference_time, detected_class_frame = stream_video(args.video, inference_engine, args.size, classes_labels, args.encoder, args.threshold, args.show_stream, args.save_video, args.show_name, args.show_confidence)
    if args.print_format in ['', 'empty']:
        print('Frame count:', frame_count)
        print('Eplased time: ', f'{eplased_time:.4f}s')
        print('Average FPS (with preprocessing and postprocessing):', fps)
        print('Average inference time per seconds: ', f'{(avg_inference_time):.4f}s')    
        print('Class detected frame count:')
        print(json.dumps(detected_class_frame, indent=4))
    if args.print_format=='json':
        output_toprint = {
            'model': args.model,
            'video': args.video,
            'frame_count': frame_count, 
            'eplased_time': f'{eplased_time:.4f}s',
            'avg_fps': fps,
            'avg_inference_time': avg_inference_time,
            'class_detected_frame_count': detected_class_frame
        }
        print(json.dumps(output_toprint), indent=4)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str, default='bicycle_thief.mp4')
    parser.add_argument('--engine', '-e', type=str, default='torch', help='torch or onnx')
    parser.add_argument('--model', '-m', type=str, default='rtdetr_yolov9bb_ep27.onnx')
    parser.add_argument('--model-conf', '-mc', type=str, default='rtdetr_cyolov9ebb_L_cocotrimmed.yml')
    parser.add_argument('--threshold', '-t', type=float, default=0.7)
    parser.add_argument('--show-stream', '-ss', action='store_true', default=False)
    parser.add_argument('--size', '-s', type=int, default=640)
    parser.add_argument('--save-video', '-sv', type=str, default='', help='must mkv or set empty ('') to not save the output')
    parser.add_argument('--print-format', '-pf', type=str, default='', help='empty or json')
    parser.add_argument('--encoder', type=str, default='XVID', help='XVID MJPG MPEG')
    parser.add_argument('--classes-labels', '-c', type=str, default='inference_class_labels', help='name_label: index_label | by json path or the json string')
    parser.add_argument('--show-name', '-sn', action='store_true', default=False, help='Show object name on the top of the bounding box')
    parser.add_argument('--show-confidence', '-sc', action='store_true', default=False, help='Show object prediction confidence on the top of the bounding box')
    args = parser.parse_args()

    main(args)

# python3 tools/inference_video.py --model=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/rtdetr_yolov9ebb_ep27.pth --engine=torch --video=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/bicycle_thief.mp4 --save-video=out.mkv --size=640 --model-conf=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_cyolov9ebb_L_cocotrimmed.yml --show-name --show-confidence

# python tools/inference_video.py --model="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\rtdetr_yolov9ebb_ep27.pth" --engine=torch --video="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\bicycle_thief.mp4" --save-video=out.mkv --size=640 --model-conf="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\configs\rtdetr\rtdetr_cyolov9ebb_L_cocotrimmed.yml" --classes-labels="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\tools\inference_class_labels.json" --show-stream --show-name --show-confidence