from Architectures.resnet import resnet18
from Architectures.mobilenetv2 import MobileNetV2
from Architectures.mobilenetv3 import mobilenet_v3_small

ls = [resnet18(num_classes=10, perforation_mode=(1,1)), MobileNetV2(num_classes=10, perforation_mode=(1,1)),
      mobilenet_v3_small(num_classes=10, perforation_mode=(1,1))]

for l in ls:
    print(l._get_n_calc())