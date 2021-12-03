#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#define DEVICE 0
#define BATCH_SIZE 1
#define CONF_THRESH 0.5
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int CHANNEL = 64;
static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int OUTPUT_SIZE =512*512;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;



std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    //std::cout << "len " << len << std::endl;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;

}


IActivationLayer* basicBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    std::cout << *(float*)(weightMap[lname + "conv1.weight"].values) << std::endl;
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1,1 });
    conv2->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{ stride, stride });
        conv3->setPaddingNd(DimsHW{ 0,0 });
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    assert(ew1);
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}


IActivationLayer* basicBlock_deeplab(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride,int pad, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ pad, pad });
    conv1->setDilationNd(DimsHW{ 2,2 });
    std::cout << *(float*)(weightMap[lname + "conv1.weight"].values) << std::endl;
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ stride,stride });
    conv2->setPaddingNd(DimsHW{ pad, pad });
    conv2->setDilationNd(DimsHW{ 2,2 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{ stride, stride });
        conv3->setPaddingNd(DimsHW{ 0,0 });
        conv3->setDilationNd(DimsHW{ 2,2 });
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    assert(ew1);
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}


IIdentityLayer* decoder(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3,3 }, weightMap[lname + "conv1.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride,stride });
    conv1->setPaddingNd(DimsHW{ 1,1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv1.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IIdentityLayer* identity1 = network->addIdentity(*relu1->getOutput(0));
    assert(identity1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*identity1->getOutput(0), outch, DimsHW{ 3,3 }, weightMap[lname + "conv2.0.weight"], emptywts);
    assert(conv2);

    conv2->setStrideNd(DimsHW{ stride,stride });
    conv2->setPaddingNd(DimsHW{ 1,1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "conv2.1", 1e-5);

    

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
 
    IIdentityLayer* identity2 = network->addIdentity(*relu2->getOutput(0));
    assert(identity2);

    return identity2;



    

}

IResizeLayer* upsample(INetworkDefinition* network,ITensor& input,int type,float scale) {
    //type 0-nearest 1billinear 
    IResizeLayer* upsample1 = network->addResize(input);
    assert(upsample1);

    //upsample1->setResizeMode(ResizeMode::kNEAREST);
    if (type == 0) {
        upsample1->setResizeMode(ResizeMode::kNEAREST);
        upsample1->setAlignCorners(true);
        float scales[] = { 1.0,2.0,2.0 };
        upsample1->setScales(scales, 3);
    }
    else
    {
        //deeplabv3p
        upsample1->setResizeMode(ResizeMode::kLINEAR);
        upsample1->setAlignCorners(false);
        float scales[] = { 1,scale,scale };
        upsample1->setScales(scales, 3);
    }
    
    
    assert(upsample1);
    return upsample1;

}


IIdentityLayer* cat(INetworkDefinition* network, ITensor& input0,ITensor& input1) {

    IResizeLayer* upsample1 = network->addResize(input0);
    assert(upsample1);

    upsample1->setResizeMode(ResizeMode::kNEAREST);
    upsample1->setAlignCorners(true);
    float scales[] = { 1.0,2.0,2.0 };
    upsample1->setScales(scales, 3);

    ITensor* inputTensors[] = { upsample1->getOutput(0),&input1 };
    IConcatenationLayer* cat = network->addConcatenation(inputTensors, 2);


    IIdentityLayer* identity = network->addIdentity(*cat->getOutput(0));
    assert(identity);
    return identity;
}



IIdentityLayer* assp(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input0, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input0, 256, DimsHW{ 1,1 }, weightMap[lname + "convs.0.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1,1});
    conv1->setPaddingNd(DimsHW{ 0,0 });
    IScaleLayer* bn1 = addBatchNorm2d(network,weightMap,*conv1->getOutput(0),lname+"convs.0.1",1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);//layer1

    IConvolutionLayer* conv2 = network->addConvolutionNd(input0,512, DimsHW{ 3,3 }, weightMap[lname + "convs.1.0.0.weight"], emptywts);
    assert(conv2);

    conv2->setDilationNd(DimsHW{ 12,12 });
    conv2->setStrideNd(DimsHW{ 1,1 });
    conv2->setPaddingNd(DimsHW{ 12,12 });
    conv2->setNbGroups(512);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*conv2->getOutput(0), 256, DimsHW{ 1,1 }, weightMap[lname + "convs.1.0.1.weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{ 1,1 });
    conv3->setPaddingNd(DimsHW{ 0,0 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "convs.1.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);//layer2


    IConvolutionLayer* conv4 = network->addConvolutionNd(input0, 512, DimsHW{ 3,3 }, weightMap[lname + "convs.2.0.0.weight"], emptywts);
    assert(conv4);

    conv4->setDilationNd(DimsHW{ 24,24 });
    conv4->setStrideNd(DimsHW{ 1,1 });
    conv4->setPaddingNd(DimsHW{ 24,24 });
    conv4->setNbGroups(512);
    IConvolutionLayer* conv5 = network->addConvolutionNd(*conv4->getOutput(0), 256, DimsHW{ 1,1 }, weightMap[lname + "convs.2.0.1.weight"], emptywts);
    assert(conv5);
    conv5->setStrideNd(DimsHW{ 1,1 });
    conv5->setPaddingNd(DimsHW{ 0,0 });
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), lname + "convs.2.1", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);//layer3

    IConvolutionLayer* conv6 = network->addConvolutionNd(input0, 512, DimsHW{ 3,3 }, weightMap[lname + "convs.3.0.0.weight"], emptywts);
    assert(conv6);
    conv6->setDilationNd(DimsHW{ 36,36 });
    conv6->setStrideNd(DimsHW{ 1,1 });
    conv6->setPaddingNd(DimsHW{ 36,36 });
    conv6->setNbGroups(512);
    IConvolutionLayer* conv7 = network->addConvolutionNd(*conv6->getOutput(0), 256, DimsHW{ 1,1 }, weightMap[lname + "convs.3.0.1.weight"], emptywts);
    assert(conv7);
    conv7->setStrideNd(DimsHW{ 1,1 });
    conv7->setPaddingNd(DimsHW{ 0,0 });
    IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv7->getOutput(0), lname + "convs.3.1", 1e-5);
    IActivationLayer* relu4 = network->addActivation(*bn4->getOutput(0), ActivationType::kRELU);//layer4


    IPoolingLayer* pool1 = network->addPoolingNd(input0, PoolingType::kAVERAGE, DimsHW{ 1,1 });

    IConvolutionLayer* conv8 = network->addConvolutionNd(*pool1->getOutput(0), 256, DimsHW{ 1,1 }, weightMap[lname + "convs.4.1.weight"], emptywts);
    assert(conv8);

    conv8->setStrideNd(DimsHW{ 1,1 });
    conv8->setPaddingNd(DimsHW{ 0,0 });

    IScaleLayer* bn5 = addBatchNorm2d(network, weightMap, *conv8->getOutput(0), lname + "convs.4.2",1e-5);

    IActivationLayer* relu5 = network->addActivation(*bn5->getOutput(0), ActivationType::kRELU);

    IResizeLayer* resize1 = upsample(network, *relu5->getOutput(0),1,1);
    

    ITensor* inputTensors[] = { relu1->getOutput(0),relu2->getOutput(0),relu3->getOutput(0) ,relu4->getOutput(0),resize1->getOutput(0)};
    IConcatenationLayer* cat = network->addConcatenation(inputTensors, 5);
    
    IIdentityLayer* identity0 = network->addIdentity(*cat->getOutput(0));
    assert(identity0);
    return identity0;

    
}


ICudaEngine* createEngine_unet_resnet34(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../unet_resnet34.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IIdentityLayer* identity =network->addIdentity(*data);
    assert(identity);
    IConvolutionLayer* conv1 = network->addConvolutionNd(*identity->getOutput(0), 64, DimsHW{ 7,7 }, weightMap["encoder.conv1.weight"], emptywts);
    
    /*IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 7,7 }, weightMap["encoder.conv1.weight"],emptywts);*/
     
    

    conv1->setStrideNd(DimsHW{ 2,2 });

    conv1->setPaddingNd(DimsHW{ 3,3 });
    assert(conv1);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "encoder.bn1", 1e-5);
    
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    assert(relu1);

    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    assert(pool1);

    pool1->setStrideNd(DimsHW(2, 2));
    pool1->setPaddingNd(DimsHW(1, 1));

    //encoder
    //layer 1
    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64,1,"encoder.layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64,1,"encoder.layer1.1.");
    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 64,1,"encoder.layer1.2.");

    //layer 2
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 64, 128, 2, "encoder.layer2.0.");
    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 128, 1, "encoder.layer2.1.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 128, 128, 1, "encoder.layer2.2.");
    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 128, 128, 1, "encoder.layer2.3.");
    


    //layer 3
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 128, 256, 2, "encoder.layer3.0.");
    IActivationLayer* relu10 = basicBlock(network, weightMap, *relu9->getOutput(0), 256, 256, 1, "encoder.layer3.1.");
    IActivationLayer* relu11 = basicBlock(network, weightMap, *relu10->getOutput(0), 256, 256, 1, "encoder.layer3.2.");
    IActivationLayer* relu12 = basicBlock(network, weightMap, *relu11->getOutput(0), 256, 256, 1, "encoder.layer3.3.");
    IActivationLayer* relu13 = basicBlock(network, weightMap, *relu12->getOutput(0), 256, 256, 1, "encoder.layer3.4.");
    IActivationLayer* relu14 = basicBlock(network, weightMap, *relu13->getOutput(0), 256, 256, 1, "encoder.layer3.5.");

    //layer4
    IActivationLayer* relu15 = basicBlock(network, weightMap, *relu14->getOutput(0), 256, 512, 2, "encoder.layer4.0.");
    IActivationLayer* relu16 = basicBlock(network, weightMap, *relu15->getOutput(0), 512, 512, 1, "encoder.layer4.1.");
    IActivationLayer* relu17 = basicBlock(network, weightMap, *relu16->getOutput(0), 512, 512, 1, "encoder.layer4.2.");

    //decoder
    IIdentityLayer* identity0 = cat(network, *relu17->getOutput(0), *relu14->getOutput(0));//512+256
    IIdentityLayer* identity1 = decoder(network, weightMap, *identity0->getOutput(0), 768, 256, 1, "decoder.blocks.0.");//256 32 32
    IIdentityLayer* identity2 = cat(network,*identity1->getOutput(0),*relu8->getOutput(0));//256+128
    IIdentityLayer* identity3 = decoder(network, weightMap, *identity2->getOutput(0), 384, 128, 1, "decoder.blocks.1.");
    IIdentityLayer* identity4 = cat(network, *identity3->getOutput(0), *relu4->getOutput(0));//128+64
    IIdentityLayer* identity5 = decoder(network, weightMap, *identity4->getOutput(0), 192, 64, 1, "decoder.blocks.2.");
    IIdentityLayer* identity6 = cat(network, *identity5->getOutput(0), *relu1->getOutput(0));//64+64
    IIdentityLayer* identity7 = decoder(network, weightMap, *identity6->getOutput(0), 128, 32, 1, "decoder.blocks.3.");
    IResizeLayer* resize1 = upsample(network, *identity7->getOutput(0),0,2);
    IIdentityLayer* identity8 = decoder(network, weightMap, *resize1->getOutput(0), 32, 16, 1, "decoder.blocks.4.");


    ////head
    IConvolutionLayer* conv2 = network->addConvolutionNd(*identity8->getOutput(0), 1, DimsHW{ 3,3 }, weightMap["segmentation_head.0.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1,1 });
    conv2->setPaddingNd(DimsHW{ 1,1 });
    conv2->setBiasWeights(weightMap["segmentation_head.0.bias"]);
    IIdentityLayer* identity9 = network->addIdentity(*conv2->getOutput(0));

    //IActivationLayer* sigmoid1 = network->addActivation(*identity9->getOutput(0), ActivationType::kSIGMOID);

    //sigmoid1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    //network->markOutput(*sigmoid1->getOutput(0));
    identity9->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*identity9->getOutput(0));
    //build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

    std::cout << "building engine ,please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine successfully" << std::endl;
    network->destroy();
    //release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;

    
}

ICudaEngine* createEngine_deeplabv3_resnet34(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../unet_deeplabv3p.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IIdentityLayer* identity = network->addIdentity(*data);
    assert(identity);
    IConvolutionLayer* conv1 = network->addConvolutionNd(*identity->getOutput(0), 64, DimsHW{ 7,7 }, weightMap["encoder.conv1.weight"], emptywts);


    conv1->setStrideNd(DimsHW{ 2,2 });

    conv1->setPaddingNd(DimsHW{ 3,3 });
    assert(conv1);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "encoder.bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    assert(relu1);

    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3,3 });
    assert(pool1);

    pool1->setStrideNd(DimsHW(2, 2));
    pool1->setPaddingNd(DimsHW(1, 1));

    //encoder
    //layer 1
    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "encoder.layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "encoder.layer1.1.");
    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 64, 1, "encoder.layer1.2.");//layer1 end

    //layer 2
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 64, 128, 2, "encoder.layer2.0.");
    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 128, 1, "encoder.layer2.1.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 128, 128, 1, "encoder.layer2.2.");
    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 128, 128, 1, "encoder.layer2.3.");//layer2 end



    //layer 3
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 128, 256, 2, "encoder.layer3.0.");
    IActivationLayer* relu10 = basicBlock(network, weightMap, *relu9->getOutput(0), 256, 256, 1, "encoder.layer3.1.");
    IActivationLayer* relu11 = basicBlock(network, weightMap, *relu10->getOutput(0), 256, 256, 1, "encoder.layer3.2.");
    IActivationLayer* relu12 = basicBlock(network, weightMap, *relu11->getOutput(0), 256, 256, 1, "encoder.layer3.3.");
    IActivationLayer* relu13 = basicBlock(network, weightMap, *relu12->getOutput(0), 256, 256, 1, "encoder.layer3.4.");
    IActivationLayer* relu14 = basicBlock(network, weightMap, *relu13->getOutput(0), 256, 256, 1, "encoder.layer3.5.");//layer3 end

    //layer4
    IActivationLayer* relu15 = basicBlock_deeplab(network, weightMap, *relu14->getOutput(0), 256, 512, 1,2, "encoder.layer4.0.");
    IActivationLayer* relu16 = basicBlock_deeplab(network, weightMap, *relu15->getOutput(0), 512, 512, 1,2, "encoder.layer4.1.");
    IActivationLayer* relu17 = basicBlock_deeplab(network, weightMap, *relu16->getOutput(0), 512, 512, 1,2, "encoder.layer4.2.");//layer4 end

    //decoder
    IIdentityLayer* identity0 = assp(network, weightMap, *relu17->getOutput(0),256, "decoder.aspp.0.");//1280
    IConvolutionLayer* conv2 = network->addConvolution(*identity0->getOutput(0), 256, DimsHW{ 1,1 }, weightMap["decoder.aspp.0.project.0.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1,1 });
    conv2->setPaddingNd(DimsHW{ 0,0 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "decoder.aspp.0.project.1",1e-5);
    IActivationLayer* relu18 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    
    IConvolutionLayer* conv3 = network->addConvolution(*relu18->getOutput(0), 256, DimsHW{ 3,3 }, weightMap["decoder.aspp.1.0.weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{ 1,1 });
    conv3->setPaddingNd(DimsHW{ 1,1 });
    conv3->setNbGroups(256);


    IConvolutionLayer* conv4 = network->addConvolution(*conv3->getOutput(0), 256, DimsHW{ 1,1 }, weightMap["decoder.aspp.1.1.weight"], emptywts);
    assert(conv4);
    conv4->setStrideNd(DimsHW{ 1,1 });
    conv4->setPaddingNd(DimsHW{ 0,0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), "decoder.aspp.2", 1e-5);

    IActivationLayer* relu19 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);
    assert(relu19);

    IResizeLayer* resize1 = upsample(network, *relu19->getOutput(0), 1,4);
    assert(resize1);

    IConvolutionLayer* conv5 = network->addConvolution(*relu4->getOutput(0), 48, DimsHW{ 1,1 }, weightMap["decoder.block1.0.weight"], emptywts);
    assert(conv5);
    conv5->setStrideNd(DimsHW{ 1,1 });
    conv5->setPaddingNd(DimsHW{ 0,0 });

    IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), "decoder.block1.1", 1e-5);
    
    IActivationLayer* relu20 = network->addActivation(*bn4->getOutput(0), ActivationType::kRELU);
    assert(relu20);

    
    ITensor* inputTensors[] = { resize1->getOutput(0),relu20->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 2);
    assert(cat1);

    IConvolutionLayer* conv6 = network->addConvolution(*cat1->getOutput(0), 304, DimsHW{ 3,3 }, weightMap["decoder.block2.0.0.weight"], emptywts);
    assert(conv6);
    conv6->setStrideNd(DimsHW{ 1,1 });
    conv6->setPaddingNd(DimsHW{ 1,1 });
    conv6->setNbGroups(304);

    IConvolutionLayer* conv7 = network->addConvolution(*conv6->getOutput(0), 256, DimsHW{ 1,1 }, weightMap["decoder.block2.0.1.weight"], emptywts);
    assert(conv7);
    conv7->setStrideNd(DimsHW{ 1,1 });
    conv7->setPaddingNd(DimsHW{ 0,0 });
    
    IScaleLayer* bn5 = addBatchNorm2d(network, weightMap, *conv7->getOutput(0), "decoder.block2.1", 1e-5);

    IActivationLayer* relu21 = network->addActivation(*bn5->getOutput(0), ActivationType::kRELU);
    assert(relu21);


    IConvolutionLayer* conv8 = network->addConvolution(*relu21->getOutput(0), 1, DimsHW{ 1,1 }, weightMap["segmentation_head.0.weight"], emptywts);
    assert(conv8);
    conv8->setStrideNd(DimsHW{ 1,1 });
    conv8->setPaddingNd(DimsHW{ 0,0 });
    conv8->setBiasWeights(weightMap["segmentation_head.0.bias"]);

    IResizeLayer* resize2 = upsample(network, *conv8->getOutput(0), 1, 4);
    assert(resize2);
    //IIdentityLayer* identity0 = cat(network, *relu17->getOutput(0), *relu14->getOutput(0));//512+256
    //IIdentityLayer* identity1 = decoder(network, weightMap, *identity0->getOutput(0), 768, 256, 1, "decoder.blocks.0.");//256 32 32
    //IIdentityLayer* identity2 = cat(network, *identity1->getOutput(0), *relu8->getOutput(0));//256+128
    //IIdentityLayer* identity3 = decoder(network, weightMap, *identity2->getOutput(0), 384, 128, 1, "decoder.blocks.1.");
    //IIdentityLayer* identity4 = cat(network, *identity3->getOutput(0), *relu4->getOutput(0));//128+64
    //IIdentityLayer* identity5 = decoder(network, weightMap, *identity4->getOutput(0), 192, 64, 1, "decoder.blocks.2.");
    //IIdentityLayer* identity6 = cat(network, *identity5->getOutput(0), *relu1->getOutput(0));//64+64
    //IIdentityLayer* identity7 = decoder(network, weightMap, *identity6->getOutput(0), 128, 32, 1, "decoder.blocks.3.");
    //IResizeLayer* resize1 = upsample(network, *identity7->getOutput(0),0);
    //IIdentityLayer* identity8 = decoder(network, weightMap, *resize1->getOutput(0), 32, 16, 1, "decoder.blocks.4.");


    ////head
    /*IConvolutionLayer* conv2 = network->addConvolutionNd(*identity8->getOutput(0), 1, DimsHW{ 3,3 }, weightMap["segmentation_head.0.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1,1 });
    conv2->setPaddingNd(DimsHW{ 1,1 });
    IIdentityLayer* identity9 = network->addIdentity(*conv2->getOutput(0));*/

    //IActivationLayer* sigmoid1 = network->addActivation(*identity9->getOutput(0), ActivationType::kSIGMOID);

    //sigmoid1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    //network->markOutput(*sigmoid1->getOutput(0));
    resize2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*resize2->getOutput(0));
    //build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

    std::cout << "building engine ,please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine successfully" << std::endl;
    network->destroy();
    //release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;


}


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builderr
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    //ICudaEngine* engine = createEngine_unet_resnet34(maxBatchSize, builder, config, DataType::kFLOAT);
    ICudaEngine* engine = createEngine_deeplabv3_resnet34(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //流同步：通过cudaStreamSynchronize()来协调。
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}
cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


struct  Detection {
    float mask[INPUT_W * INPUT_H * 1];
};

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

void process_cls_result(Detection& res, float* output) {
    for (int i =0; i < OUTPUT_SIZE; i++) {
        //std::cout << *(output + i) << std::endl;
        res.mask[i] = sigmoid(*(output + i));
        
    }
}

int build() {
    IHostMemory* modelStream{ nullptr };
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);
    std::string engine_name = "unet_deeplabv3p.engine";
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
}

//int main() {
//    build();
//}


int detect() {
    cudaSetDevice(DEVICE);
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    //std::string engine_name = "unet_resnet34.engine";
    std::string engine_name = "deeplabv3_resnet34.engine";
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    const char* path = "C:\\workspace\\tensorrtx-master\\resnet\\build\\resnet\\pic\\";
    std::vector<std::string> file_names;
    if (read_files_in_dir(path, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;



    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(std::string(path) + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB

            /*for (int i = 0; i < INPUT_H; i++) {

                for (int j = 0; j < INPUT_W; j++) {
                   int a= pr_img.at<cv::Vec3b>(j, i)[0];
                   int b = pr_img.at<cv::Vec3b>(j, i)[1];
                   int c = pr_img.at<cv::Vec3b>(j, i)[2];
                   std::cout << a << std::endl;
                   std::cout << b << std::endl;
                   std::cout << c << std::endl;
                }
            }*/

            //cv::normalize(pr_img, pr_img,{0.485, 0.456, 0.406 }, { 0.229,0.224,0.225 });
            //cv::normalize(pr_img, pr_img, 1, 0);
            // cv::imwrite("s_o" + file_names[f - fcount + 1 + b] + "_unet.jpg", pr_img); 
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {

                    data[b * 3 * INPUT_H * INPUT_W + i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
                    /* std::cout << ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229 << std::endl;
                     std::cout << ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224 << std::endl;
                     std::cout << ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225 << std::endl;*/
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;



        std::vector<Detection> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            process_cls_result(res, &prob[b * OUTPUT_SIZE]);
        }

        std::cout << fcount << std::endl;

        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            float* mask = res.mask;
            cv::Mat mask_mat = cv::Mat(INPUT_H, INPUT_W, CV_8UC1, cv::Scalar(0));
            //cv::imwrite("2.bmp", mask_mat);
            for (int i = 0; i < INPUT_H; i++) {
                //ptmp = mask_mat.ptr<uchar>(i);
                for (int j = 0; j < INPUT_W; j++) {
                    float* pixcel = mask + i * INPUT_W + j;
                    
                    
                    //mask_mat.at<uchar>(j, i) = 255;
                    if (*pixcel >CONF_THRESH) {
                        mask_mat.at<uchar>(j, i) = 255;

                    }
                    else
                    {
                        mask_mat.at<uchar>(j, i) = 0;

                    }

                }
            }

            /*for (int i = 0; i < INPUT_H; i++) {
                uchar* data = mask_mat.ptr<uchar>(i);
                for (int j = 0; j < INPUT_W; j++) {
                    if ((int)data[j] > 0) {
                        std::cout << (int)data[j] <<std::endl;
                   }

                }
            }*/

            cv::imwrite("s_" + file_names[f - fcount + 1 + b] + "_unet.jpg", mask_mat);


        }
        fcount = 0;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

int main() {

    //build();
    detect();

    return 0;
}



