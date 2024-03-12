from commons_pt import *
from src.core import register

__all__ = ['YoloV9Backbone']

@register
class YoloV9Backbone(nn.Module):
    def __init__(self, c=[3, 64, 128, 256, 512, 1024, 2048], return_idx=[2,3,4], device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.return_idx = return_idx

        # 3-> 64
        silence = Silence()
        conv0 = Conv(c[0], c[1], 3, 2)

        # 64
        conv1 = Conv(c[1], c[2], 3, 2)
        rncelan0 = RepNCSPELAN4(c[2], c[3], c[2], c[1], 1)

        conv2 = Conv(c[3], c[4], 3, 2)
        rncelan1 = RepNCSPELAN4(c[4], c[4], c[4], c[2], 1)

        conv3 = Conv(c[4], c[5], 3, 2)
        rncelan2 = RepNCSPELAN4(c[5], c[5], c[5], c[2], 1)

        conv4 = Conv(c[5], c[6], 3, 2)
        rncelan3 = RepNCSPELAN4(c[6], c[6], c[6], c[2], 1)
        
        pyramids = [
            nn.Sequential(silence, conv0),
            nn.Sequential(conv1, rncelan0),
            nn.Sequential(conv2, rncelan1),
            nn.Sequential(conv3, rncelan2),
            nn.Sequential(conv4, rncelan3),
        ]
        self.pyramids = [pyr.to(device) for pyr in pyramids]
        # self.yolov9bb_layers = nn.Sequential(self.pyramid0, self.pyramid1, self.pyramid2, self.pyramid3, self.pyramid4)
        self.to(device)

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