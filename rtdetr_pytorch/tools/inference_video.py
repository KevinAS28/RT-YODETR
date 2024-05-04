import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import time
import argparse
import json

import cv2
import torch
from torchvision.transforms import v2 as trfmv2
import numpy as np

from tools.inf_utils import *

def line_to_box(line, img_shape, line_type='h', invert=False):
  line = list(line)
  if line_type=='h':
    if not invert: # [0, 400]
      line = [0, line[1], img_shape[0], img_shape[1]] # [0, 400, 640, 640]
    else: # [0, 400]
      line = [0, line[0], img_shape[0], line[1]] # [0, 0, 400, 640]
  elif line_type=='v':
    if not invert: # [400, 0]
      line = [line[0], 0, img_shape[0], img_shape[1]] # [400, 0, 640, 640]
    else:
      line = [0, 0, line[0], img_shape[1]] # [0, 0, 400, 640]
  else:
    raise ValueError(f'Line type {line_type} is not supported')  
  
  x1, y1, x2, y2 = line # 640, 0, 0, 640 | 0, 300, 200, 0
  if x1>x2 or y1>y2:
    return [x2, y2, x1, y1]  # Swap coords
  return line
                               
def rectangles_intersect(rect0, rect1, invert=False):
  
  if (rect0[0]**2+rect0[1]**2)**(1/2) < (rect1[0]**2+rect1[1]**2)**(1/2):
    conditions = (
      rect0[2]>=rect1[0] and rect0[3]>=rect1[1],
      rect1[2]>=rect0[0] and rect1[3]<=rect0[1],  
    )
  else:
    conditions = (
      rect1[2]>=rect0[0] and rect1[3]>=rect0[3],
      rect1[3]>=rect0[1] and rect1[2]>=rect0[0],
    )
  result = any(conditions)

  return not result if invert else result

def is_bbox_intersection(bbox1, bbox2):
  if bbox2[0] > bbox1[2] or bbox1[0] > bbox2[2]:
    return False
  if bbox2[1] > bbox1[3] or bbox1[1] > bbox2[3]:
    return False 
  return True

def obj_crossed_line(obj_bbox, line, line_type='h', invert=False):
    '''
        DEPRECATED, NEXT TIME USE is_bbox_intersection()
    '''
    def _algo():
        points = [obj_bbox[:2], obj_bbox[2:4], [obj_bbox[2], obj_bbox[1]], [obj_bbox[0], obj_bbox[3]]]

        # check vertical
        if line_type=='v':
            for pt in points:
                if pt[0]>line[0]:
                    return True
            return False
        
        elif line_type=='h':
            for pt in points:
                if pt[1]>line[1]:
                    return True
        
            return False
        
        else:
            raise 'line not supported'

    return _algo() and not invert

def add_overlay(img, rect, channel_index, color_value, alpha=0.5, invert=False):
    start_w, start_h, end_w, end_h = rect
    height, width, channels = img.shape

    new_colors = [0,0,0,alpha*255]
    new_colors[channel_index] = color_value

    if invert:
        print(height, width, start_h, end_h, start_w, end_w)
        overlay = np.full((height, width, 4), new_colors, dtype=np.uint8)
        overlay_rgb = overlay[..., :channels] 
        mask = np.ones_like(img, dtype=bool)
        mask[start_h:end_h, start_w:end_w] = False
        img = (1 - alpha) * img + alpha * overlay_rgb * mask.astype(np.float32)
        img = img.astype(np.uint8)
    else:
        overlay = np.full((end_h - start_h, end_w - start_w, 4), new_colors, dtype=np.uint8)
        overlay_rgb = overlay[..., :channels] 
        img[start_h:end_h, start_w:end_w] = (1 - alpha) * img[start_h:end_h, start_w:end_w] + alpha * overlay_rgb

    return img


def stream_video(video_path, inference_engine, size, classes_labels, encoder='XVID', thrh=0.65, show_stream=False, out_video_path='', draw_obj_name=True, draw_obj_conf=True, additional_postprocessor=lambda frame: frame, obj_warning=lambda s,l,b: None):
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
            postprocessed_frame = additional_postprocessor(postprocessed_frame)

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
                    obj_warning(s, lab_str, b)
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

def frame_overlay(lines, frame, invert=False):
    for ln in lines:
        ln_type, ln = ln[0], ln[1:]
        ln_bx = line_to_box(ln, frame.shape, ln_type, False)
        frame = add_overlay(frame, ln_bx, 0, 255, 0.25, invert)

    return frame

def object_warnings(lines, s, l, b, objects_to_warn=['person'], invert=False):
    for ln in lines:
        ln_type, ln = ln[0], ln[1:]    
        obj_crossed = False
        if l in objects_to_warn:
            obj_crossed = obj_crossed_line(b, ln, ln_type, invert)
            if obj_crossed:
                print(f'WARNING: OBJECT {l} HAS CROSSED THE LINE')      

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

    if len(args.overlay)>0:
        # lines = [('h', 0, 450, 640, 450)]
        # invert = False        
        line = args.overlay.split('_')
        line_coord = [int(i) for i in line[1:5]]
        line_invert = bool(int(line[-1]))
        line_orientation = line[0]
        lines = [[line_orientation, *line_coord]]
        invert = line_invert

        postprocessor = lambda frame: frame_overlay(lines, frame, invert)
        obj_warn = lambda s,l,b: object_warnings(lines, s, l, b, objects_to_warn=['person'], invert=invert)
    else:
       postprocessor = lambda frame: frame
       obj_warn = lambda s,l,b: None    

    frame_count, eplased_time, fps, avg_inference_time, detected_class_frame = stream_video(args.video, inference_engine, args.size, classes_labels, args.encoder, args.threshold, args.show_stream, args.save_video, args.show_name, args.show_confidence, postprocessor, obj_warn)
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
    parser.add_argument('--overlay', '-ol', type=str, default='h_0_450_640_450_0', help='Line overlay to detect any objects crossing the line format: (line orientation v|h)_startx_starty_endx_endy_(invert 0|1). Example: --overlay=h_0_450_640_450_0')
    args = parser.parse_args()

    main(args)

# python3 tools/inference_video.py --model=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/rtdetr_yolov9ebb_ep27.pth --engine=torch --video=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/bicycle_thief.mp4 --save-video=out.mkv --size=640 --model-conf=/home/kevin/Custom-RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_cyolov9ebb_L_cocotrimmed.yml --classes-labels='/home/kevin/Custom-RT-DETR/rtdetr_pytorch/tools/inference_class_labels.json' --show-name --show-confidence 

# python tools/inference_video.py --model="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\rtdetr_yolov9ebb_ep27.pth" --engine=torch --video="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\bicycle_thief.mp4" --save-video=out.mkv --size=640 --model-conf="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\configs\rtdetr\rtdetr_cyolov9ebb_L_cocotrimmed.yml" --classes-labels="C:\Users\kevin\Documents\Custom-RT-DETR\rtdetr_pytorch\tools\inference_class_labels.json" --show-stream --show-name --show-confidence --overlay=h_0_450_640_450_0