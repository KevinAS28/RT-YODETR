import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import subprocess
import json

import torch
from torch import nn
import onnxruntime as ort

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