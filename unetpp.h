#include <map>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include<dirent.h>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
using namespace nvinfer1;
using namespace std;

static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int OUTPUT_SIZE = 1 * 512*512;
static const int BATCH_SIZE = 1;



const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

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

static Logger gLogger;
IExecutionContext* context;
IRuntime* runtime;
ICudaEngine* engine;
void* buffers[2];

int inputIndex;
int outputIndex;
cudaStream_t stream;

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



IResizeLayer* upsample(INetworkDefinition* network, ITensor& input, int type, float scale) {
	//type 0-nearest 1billinear 
	IResizeLayer* upsample1 = network->addResize(input);
	assert(upsample1);

	//upsample1->setResizeMode(ResizeMode::kNEAREST);
	if (type == 0) {
		upsample1->setResizeMode(ResizeMode::kNEAREST);
		upsample1->setAlignCorners(true);
		float scales[] = { 1.0,scale,scale };
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

IActivationLayer* decoder_block(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int stride, std::string lname) {
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + ".0.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ stride, stride });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);
	return relu1;

}

IActivationLayer* upsample_decoder_block(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input1,ITensor& input2, int outch, int stride, std::string lname1,std::string lname2) {

	IResizeLayer* resize = upsample(network, input1, 0, 2);
	ITensor* inputTensor[] = { resize->getOutput(0),&input2 };
	IConcatenationLayer* cat= network->addConcatenation(inputTensor, 2);
	assert(cat);

	IActivationLayer*relu1=decoder_block(network, weightMap, *cat->getOutput(0), outch, stride, lname1);
	IActivationLayer*relu2 = decoder_block(network, weightMap, *relu1->getOutput(0), outch, stride, lname2);

	
	return relu2;

}





ICudaEngine* createEngine_unetpp_resnet34(std::string wts_path, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)

{
	INetworkDefinition* network = builder->createNetworkV2(0U);

	// Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
	assert(data);

	std::map<std::string, Weights> weightMap = loadWeights(wts_path);
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
	IActivationLayer* relu15 = basicBlock(network, weightMap, *relu14->getOutput(0), 256, 512, 2, "encoder.layer4.0.");
	IActivationLayer* relu16 = basicBlock(network, weightMap, *relu15->getOutput(0), 512, 512, 1, "encoder.layer4.1.");
	IActivationLayer* relu17 = basicBlock(network, weightMap, *relu16->getOutput(0), 512, 512, 1,  "encoder.layer4.2.");//layer4 end


	
	//x_0_0 256 32 32
	//x_1_1 128 64 64
	//x_2_2 64 128 128
	//x_3_3 64 256 256
	std::map<std::string,ITensor*>dense_x;
	IActivationLayer* relu_decoderx_0_0=upsample_decoder_block(network, weightMap, *relu17->getOutput(0), *relu14->getOutput(0), 256, 1, "decoder.blocks.x_0_0.conv1", "decoder.blocks.x_0_0.conv2");
	dense_x.insert(std::pair<std::string,ITensor*>("0_0",relu_decoderx_0_0->getOutput(0)));
	IActivationLayer* relu_decoderx_1_1 = upsample_decoder_block(network, weightMap, *relu14->getOutput(0), *relu8->getOutput(0), 128, 1, "decoder.blocks.x_1_1.conv1", "decoder.blocks.x_1_1.conv2");
	dense_x.insert(std::pair<std::string, ITensor*>("1_1", relu_decoderx_1_1->getOutput(0)));
	IActivationLayer* relu_decoderx_2_2 = upsample_decoder_block(network, weightMap, *relu8->getOutput(0), *relu4->getOutput(0), 64, 1, "decoder.blocks.x_2_2.conv1", "decoder.blocks.x_2_2.conv2");
	dense_x.insert(std::pair<std::string, ITensor*>("2_2", relu_decoderx_2_2->getOutput(0)));
	IActivationLayer* relu_decoderx_3_3 = upsample_decoder_block(network, weightMap, *relu4->getOutput(0), *relu1->getOutput(0), 64, 1, "decoder.blocks.x_3_3.conv1", "decoder.blocks.x_3_3.conv2");
	dense_x.insert(std::pair<std::string, ITensor*>("3_3", relu_decoderx_3_3->getOutput(0)));

	std::vector<ITensor*>features;
	features.push_back(relu17->getOutput(0));
	features.push_back(relu14->getOutput(0));
	features.push_back(relu8->getOutput(0));
	features.push_back(relu4->getOutput(0));
	features.push_back(relu1->getOutput(0));
	
	//x_0_1
	//x_1_1 relu8 cat->cat_features
	//x_0_0 cat_features decoder_block
	std::map<std::string, int> blocks;

	blocks["0_0"] = 256;
	blocks["0_1"] = 128;
	blocks["0_2"] = 64;
	blocks["0_3"] = 32;
	blocks["0_4"] = 16;
	blocks["1_1"] = 128;
	blocks["1_2"] = 64;
	blocks["1_3"] = 64;
	blocks["2_2"] = 64;
	blocks["2_3"] = 64;
	blocks["3_3"] = 64;

	for (int layer_idx = 1; layer_idx < 4; layer_idx++) {
		for (int depth_idx = 0; depth_idx < (4 - layer_idx); depth_idx++) {
			int dense_l_i = depth_idx + layer_idx;
			std::vector<ITensor*>cat_features;
			for (int idx = depth_idx + 1; idx < dense_l_i + 1; idx++) {
				cat_features.push_back(dense_x.at(std::to_string(idx) + "_" + std::to_string(dense_l_i)));
			}
			
			if (cat_features.size() == 1) {
				ITensor* inputTensor[] = { cat_features[0],features[dense_l_i + 1] };
				IConcatenationLayer* cat = network->addConcatenation(inputTensor, 2);
				IActivationLayer* relux =upsample_decoder_block(network, weightMap,*dense_x.at(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i-1)), *cat->getOutput(0), blocks.at(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i)),1, "decoder.blocks.x_" + std::to_string(depth_idx) + "_" + std::to_string(dense_l_i) + ".conv1", "decoder.blocks.x_" + std::to_string(depth_idx) + "_" + std::to_string(dense_l_i) + ".conv2");
				dense_x.insert(std::pair<std::string, ITensor*>(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i),relux->getOutput(0)));
				
			}
			if (cat_features.size() == 2) {

				ITensor* inputTensor[] = { cat_features[0],cat_features[1],features[dense_l_i + 1] };
				IConcatenationLayer* cat = network->addConcatenation(inputTensor, 3);
				IActivationLayer* relux = upsample_decoder_block(network, weightMap, *dense_x.at(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i - 1)), *cat->getOutput(0), blocks.at(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i)), 1, "decoder.blocks.x_" + std::to_string(depth_idx) + "_" + std::to_string(dense_l_i) + ".conv1", "decoder.blocks.x_" + std::to_string(depth_idx) + "_" + std::to_string(dense_l_i) + ".conv2");
				dense_x.insert(std::pair<std::string, ITensor*>(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i), relux->getOutput(0)));

			}

			if (cat_features.size() == 3) {
				
				ITensor* inputTensor[] = { cat_features[0],cat_features[1],cat_features[2],features[dense_l_i + 1] };
				IConcatenationLayer* cat = network->addConcatenation(inputTensor, 4);
				IActivationLayer* relux = upsample_decoder_block(network, weightMap, *dense_x.at(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i-1)), *cat->getOutput(0), blocks.at(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i)), 1, "decoder.blocks.x_" + std::to_string(depth_idx) + "_" + std::to_string(dense_l_i) + ".conv1", "decoder.blocks.x_" + std::to_string(depth_idx) + "_" + std::to_string(dense_l_i) + ".conv2");
				dense_x.insert(std::pair<std::string, ITensor*>(std::to_string(depth_idx) + "_" + std::to_string(dense_l_i), relux->getOutput(0)));

			}
		}

	}

	IResizeLayer* resize_0_4 = upsample(network, *dense_x.at("0_3"), 0, 2);

	IActivationLayer*relu_0_4_1 = decoder_block(network, weightMap, *resize_0_4->getOutput(0), blocks.at("0_4"), 1, "decoder.blocks.x_0_4.conv1");
	IActivationLayer*relu_0_4_2 = decoder_block(network, weightMap, *relu_0_4_1->getOutput(0), blocks.at("0_4"), 1, "decoder.blocks.x_0_4.conv2");
	dense_x.insert(std::pair<std::string, ITensor*>("0_4", relu_0_4_2->getOutput(0)));
	//seg head

	IConvolutionLayer* conv_head = network->addConvolutionNd(*dense_x.at("0_4"), 1, DimsHW{ 3,3 }, weightMap["segmentation_head.0.weight"], emptywts);
	assert(conv_head);
	conv_head->setStrideNd(DimsHW{ 1,1 });
	conv_head->setPaddingNd(DimsHW{ 1,1 });
	conv_head->setBiasWeights(weightMap["segmentation_head.0.bias"]);
	conv_head->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*conv_head->getOutput(0));

	
	//build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(16 * (1 << 20));
#ifdef USE_FP16
	config->setFlag(BuilderFlag::kFP16);
#endif
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



void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);


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


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string wts_path)
{
	// Create builderr
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	//ICudaEngine* engine = createEngine_unet_resnet34(maxBatchSize, builder, config, DataType::kFLOAT);

	engine = createEngine_unetpp_resnet34(wts_path, maxBatchSize, builder, config, DataType::kFLOAT);
	assert(engine != nullptr);

	// Serialize the engine
	(*modelStream) = engine->serialize();

	// Close everything down
	engine->destroy();
	builder->destroy();
	config->destroy();
}

extern "C" __declspec(dllexport) int build(std::string wts_name, std::string engine_name)
{
	IHostMemory* modelStream{ nullptr };
	APIToModel(BATCH_SIZE, &modelStream, wts_name);
	assert(modelStream != nullptr);
	std::ofstream p(engine_name, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	modelStream->destroy();
	return 0;
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
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		//std::cout << *(output + i) << std::endl;
		res.mask[i] = sigmoid(*(output + i));

	}
}