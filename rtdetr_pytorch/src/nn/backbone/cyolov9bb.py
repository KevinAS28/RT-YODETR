from .commons_pt import *
from src.core import register

__all__ = ['CYoloV9Backbone']

@register
class CYoloV9Backbone(nn.Module):
    def __init__(self, return_idx=[2,3,4]):
        super().__init__()
        self.return_idx = return_idx

        self.pyramids = nn.Sequential(
            nn.Sequential(
                Silence(),
                Conv(3, 64, 3, 2),
            ),
            nn.Sequential(
                Conv(64, 128, 3, 2),
                RepNCSPELAN4(128, 256, 128, 64)
            ),
            nn.Sequential(
                ADown(256, 256),
                RepNCSPELAN4(256, 512, 256, 128, 1)
            ),            
            nn.Sequential(
                ADown(512, 1024),
                RepNCSPELAN4(1024, 1024, 1024, 256, 1)
            ),            
            nn.Sequential(  
                ADown(1024, 1024),
                RepNCSPELAN4(1024, 1024, 1024, 256, 1)
            ),  
        )            
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
    print(CYoloV9Backbone())