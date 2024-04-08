from .commons_pt import *
from src.core import register

__all__ = ['YoloV9EBackbone']

@register
class YoloV9EBackbone(nn.Module):
    def __init__(self, return_idx=[22, 25, 28], weight_path='yolov9ebb.pt', freeze=True):
        super().__init__()
        self.return_idx = return_idx

        # index, module, from_index, params
        self.layers = [(Silence(), -1, 0),
                       (Conv(3, 64, 3, 2, None, 1, 1, True), -1, 1856),
                       (Conv(64, 128, 3, 2, None, 1, 1, True), -1, 73984),
                       (RepNCSPELAN4(128, 256, 128, 64, 2), -1, 252160),
                       (ADown(c1=256, c2=256), -1, 164352),
                       (RepNCSPELAN4(256, 512, 256, 128, 2), -1, 1004032),
                       (ADown(c1=512, c2=512), -1, 656384),
                       (RepNCSPELAN4(512, 1024, 512, 256, 2), -1, 4006912),
                       (ADown(c1=1024, c2=1024), -1, 2623488),
                       (RepNCSPELAN4(1024, 1024, 512, 256, 2), -1, 4269056),
                       (CBLinear(64, [64], 1, 1, None, 1), 1, 4160),
                       (CBLinear(256, [64, 128], 1, 1, None, 1), 3, 49344),
                       (CBLinear(512, [64, 128, 256], 1, 1, None, 1), 5, 229824),
                       (CBLinear(1024, [64, 128, 256, 512], 1, 1, None, 1), 7, 984000),
                       (CBLinear(1024, [64, 128, 256, 512, 1024], 1, 1, None, 1), 9, 2033600),
                       (Conv(3, 64, 3, 2, None, 1, 1, True), 0, 1856),
                       (CBFuse([0, 0, 0, 0, 0]), [10, 11, 12, 13, 14, -1], 0),
                       (Conv(64, 128, 3, 2, None, 1, 1, True), -1, 73984),
                       (CBFuse([1, 1, 1, 1]), [11, 12, 13, 14, -1], 0),
                       (RepNCSPELAN4(128, 256, 128, 64, 2), -1, 252160),
                       (ADown(c1=256, c2=256), -1, 164352),
                       (CBFuse([2, 2, 2]), [12, 13, 14, -1], 0),
                       (RepNCSPELAN4(256, 512, 256, 128, 2), -1, 1004032),
                       (ADown(c1=512, c2=512), -1, 656384),
                       (CBFuse([3, 3]), [13, 14, -1], 0),
                       (RepNCSPELAN4(512, 1024, 512, 256, 2), -1, 4006912),
                       (ADown(c1=1024, c2=1024), -1, 2623488),
                       (CBFuse([4]), [14, -1], 0),
                       (RepNCSPELAN4(1024, 1024, 512, 256, 2), -1, 4269056)
        ]            
        model_list = []
        self.params_count = 0
        for layer in self.layers:
            module = layer[0]
            module.index_from = layer[1]
            module.params_count = layer[2]
            self.params_count += layer[2]
            model_list.append(module)
        self.model = nn.Sequential(*model_list)

        if weight_path:
            self.model.load_state_dict(torch.load(weight_path))
            print(f'weight backbone loaded from {weight_path}')
        
        if freeze:
            self.model.requires_grad_(False)
            self.requires_grad_(False)
            for p in self.model.parameters():
                p.requires_grad = False       
                     
        print('layers length:', len(self.layers))

    def forward(self, x):
        outs = []  # outputs
        for m in self.model:
            if m.index_from != -1:  # if not from previous layer
                x = outs[m.index_from] if isinstance(m.index_from, int) else [x if j == -1 else outs[j] for j in m.index_from]  # from earlier layers
            x = m(x)
            outs.append(x)
        
        return [outs[i] for i in self.return_idx]
    
if __name__=='__main__':
    model = YoloV9EBackbone()
    print(model)
    print(model.requires_grad)
    data = torch.randn(1, 3, 640, 640)
    out = model(data)
    print([i.shape for i in out])