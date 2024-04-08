import time
import argparse
import json

import cv2
import onnxruntime as ort 
import torch
from torchvision.transforms import ToTensor

print('device:', ort.get_device())

named_labels = {
    0: 'person',
    1: 'bicycle'
}

def preprocess_input(frame, size):
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_ort_session(model_path):
    providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),})]
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    ort_session.disable_fallback()
    return ort_session

def inference_imgs(ort_session, input_data):
    labels, boxes, scores = ort_session.run(None, {'images': input_data.data.numpy(), 'orig_target_sizes': torch.tensor([[640, 640]]).data.numpy()})
    return labels, boxes, scores

def stream_video(video_path, ort_session, size, thrh, show_stream=False, out_video_path=False):
    cap = cv2.VideoCapture(video_path)
    save_video_output = len(out_video_path)>0

    if save_video_output:
        fourcc_code = cv2.VideoWriter_fourcc(*'X264')
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
        name:0 for name in named_labels.values()
    }

    totensor = ToTensor()

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
                preprocessed_frame = preprocess_input(frame, size)  # Apply preprocessing
                input_data = torch.stack([totensor(preprocessed_frame)])
                labels, boxes, scores = inference_imgs(ort_session, input_data)
                inference_time = time.time()-inference_start_time
                total_inference_time += inference_time
                img_index = 0
                scr = scores[img_index]
                lab = labels[img_index][scr > thrh]

                postprocessed_frame = preprocessed_frame.copy()
                frame_count += 1
                fps = frame_count / eplased_time

                if show_stream and len(lab)>0:
                    box = [[int(j) for j in _box] for _box in boxes[0][scr > thrh]]
                    scr_str = '-'+str(round(scr[img_index]*100, 1))+'%'
                    lab_str = str(named_labels[lab[0]])+'-'

                    detected_class_frame[named_labels[lab[0]]] += 1

                    for b in box:
                        postprocessed_frame = cv2.rectangle(postprocessed_frame, tuple(b[:2]), tuple(b[2:4]), color=(0, 0, 255), thickness=2)  # Red rectangle
                        postprocessed_frame = cv2.putText(postprocessed_frame, f"{lab_str}{scr_str}", tuple(b[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text        

                postprocessed_frame = cv2.putText(postprocessed_frame, f"FPS: {fps:.2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # White text           
                eplased_time = time.time()-start_time

                if show_stream:
                    cv2.imshow("Video with Object Detection", cv2.cvtColor(postprocessed_frame, cv2.COLOR_RGB2BGR))

                if save_video_output:
                    video_writer.write(postprocessed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
    ort_session = get_ort_session(args.model)
    frame_count, eplased_time, fps, avg_inference_time, detected_class_frame = stream_video(args.video, ort_session, args.size, args.threshold, args.show_stream, args.save_video)
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
    parser.add_argument('--model', '-m', type=str, default='rtdetr_yolov9bb_ep27.onnx')
    parser.add_argument('--threshold', '-t', type=float, default=0.7)
    parser.add_argument('--show-stream', '-ss', action='store_true', default=True)
    parser.add_argument('--size', '-s', type=int, default=640)
    parser.add_argument('--save-video', '-sv', type=str, default='', help='have to be mkv. set empty to not save the output')
    parser.add_argument('--print-format', '-pf', type=str, default='', help='empty or json')
    args = parser.parse_args()

    main(args)