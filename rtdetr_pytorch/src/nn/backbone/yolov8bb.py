from commons_pt import *

from src.core import register

__all__ = ['YoloV8Backbone']

@register
class YoloV8Backbone(nn.Module):
    def __init__(self, channels=[3, 64, 256, 512, 1024, 2048], depths=[3,6,6], phi=-1, pretrained=False):
        super().__init__()
        #------------------------------------------------#
        #The input image is 3, 640, 640
        #------------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(channels[0], channels[1], 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(channels[1], channels[2], 3, 2),
            C2f(channels[2], channels[2] , depths[0], True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(channels[2], channels[3], 3, 2),
            C2f(channels[3], channels[3], depths[1], True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(channels[3] , channels[4], 3, 2),
            C2f(channels[4], channels[4], depths[2], True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(channels[4], channels[5], 3, 2),
            C2f(channels[5], channels[5], depths[0], True),
            SPPF(channels[5], channels[5], k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #------------------------------------------------#
        # The output of dark3 is 256, 80, 80, which is an effective feature layer
        #------------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #------------------------------------------------#
        # The output of dark4 is 512, 40, 40, which is an effective feature layer
        #------------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #------------------------------------------------#
        # The output of dark5 is 1024 * deep_mul, 20, 20, which is a valid feature layer
        #------------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

if __name__=='__main__':
    print(YoloV8Backbone())