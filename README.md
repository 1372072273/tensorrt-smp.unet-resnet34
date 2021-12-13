# tensorrt-smp.unet-resnet34 and smp.deeplabv3plus-resnet34 and smp.unet++-resnet34
step1:
tensorrt for segmentation models pytorch's unet and deeplabv3plus model,resnet34 is the backbone
the model is form https://github.com/qubvel/segmentation_models.pytorch,
Net type is Unet,Encoder is resnet34,you can use pip install segmentation_models.pytorch to download it,train your own model.
step2:
use the inference.py to transform the .pth model to .wts。

step3:
configure the .cpp's enviroment

step4:
use the function build to generate engine,and use the 'detect' to check the result

enviroment:
cuda 11.1
TensorRT-7.2.2.3
opencv  4.1.1



the cpp code reference https://github.com/wang-xinyu/tensorrtx/unet
