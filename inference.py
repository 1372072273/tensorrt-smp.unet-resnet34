import torch
from torch import nn
import torchvision
import os
import struct
import segmentation_models_pytorch as smp
from torchsummary import summary

def main():
    print('cuda device count: ', torch.cuda.device_count())
    state_dict = torch.load('prj1.pth')
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    CLASSES = ['ng']
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    model.load_state_dict(state_dict.state_dict())
    model.eval()
    device=torch.device("cuda")

    #return
    f = open("prj1.wts", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k,v in model.state_dict().items():
        # print('key: ', k)
        # print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()