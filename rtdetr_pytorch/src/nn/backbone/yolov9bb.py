from .commons_pt import *
from src.core import register

__all__ = ['YoloV9Backbone']

@register
class YoloV9Backbone(nn.Module):
    def __init__(self, return_idx=[2,3,4], weight_path=False, freeze=False):
        super().__init__()
        self.return_idx = return_idx

        self.pyramids = nn.Sequential(
            Silence(),
            Conv(3, 64, 3, 2),
        
            Conv(64, 128, 3, 2),
            RepNCSPELAN4(128, 256, 128, 64),
        
            ADown(256, 256),
            RepNCSPELAN4(256, 512, 256, 128, 1),
        
            ADown(512, 512),
            RepNCSPELAN4(512, 512, 512, 256, 1),
        
            ADown(512, 512),
            RepNCSPELAN4(512, 512, 512, 256, 1),
        )            
        if weight_path:
            self.pyramids.load_state_dict(torch.load(weight_path))
            print(f'weight backbone loaded from {weight_path}')
        if freeze:
            self.pyramids.requires_grad_(False)
            print(f'Backbone freezed')
            
        print('pyramids length:', len(self.pyramids))

    def forward(self, x):
        results = []
        for i in range(self.return_idx[-1]+1):
            pyr = self.pyramids[i]
            x = pyr(x)
            if i in self.return_idx:
                results.append(x)
        return results
    
if __name__=='__main__':
    print(YoloV9Backbone())