/*
 * DetectionOutputLayer.cpp
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "DetectionOutputLayer.h"
#include "BBoxUtil.h"
#include "MathFunctions.h"
#include "StdOutLog.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;
using namespace boost::property_tree;

template <typename Dtype>
DetectionOutputLayer<Dtype>::DetectionOutputLayer()
: Layer<Dtype>(),
  bboxPreds("bboxPreds"),
  bboxPermute("bboxPermute"),
  confPermute("confPermute"),
  temp("temp") {
	this->type = Layer<Dtype>::DetectionOutput;

	SASSERT(SLPROP(DetectionOutput, numClasses) > 0, "Must specify numClasses.");
	SASSERT(SLPROP(DetectionOutput, nmsParam).nmsThreshold >= 0, "nmsThreshold must be non negative.");
	SASSERT0(SLPROP(DetectionOutput, nmsParam).eta > 0.f && SLPROP(DetectionOutput, nmsParam).eta <= 1.f);

	SLPROP(DetectionOutput, saveOutputParam).resizeParam = SLPROP(DetectionOutput, resizeParam);

	this->numLocClasses = SLPROP(DetectionOutput, shareLocation) ? 1 : SLPROP(DetectionOutput, numClasses);


	if (!SLPROP(DetectionOutput, saveOutputParam).outputDirectory.empty()) {
		if (boost::filesystem::is_directory(SLPROP(DetectionOutput, saveOutputParam).outputDirectory)) {
			boost::filesystem::remove_all(SLPROP(DetectionOutput, saveOutputParam).outputDirectory);
		}
		if (!boost::filesystem::create_directories(SLPROP(DetectionOutput, saveOutputParam).outputDirectory)) {
			STDOUT_LOG("Failed to create directory: %s", SLPROP(DetectionOutput, saveOutputParam).outputDirectory.c_str());
		}
	}


	this->needSave = SLPROP(DetectionOutput, saveOutputParam).outputDirectory == "" ? false : true;
	if (SLPROP(DetectionOutput, saveOutputParam).labelMapFile != "") {
		string labelMapFile = SLPROP(DetectionOutput, saveOutputParam).labelMapFile;
		if (labelMapFile.empty()) {
			// Ignore saving if there is no labelMapFile provieded.
			STDOUT_LOG("Provide labelMapFile if output results to files.");
			this->needSave = false;
		} else {
			this->labelMap.setLabelMapPath(labelMapFile);
			this->labelMap.build();
			this->labelMap.mapLabelToName(this->labelToName);
			this->labelMap.mapLabelToName(this->labelToDisplayName);
		}
	} else {
		this->needSave = false;
	}

	if (SLPROP(DetectionOutput, saveOutputParam).nameSizeFile.empty()) {
		// Ignore saving if there is no nameSizeFile provided.
		STDOUT_LOG("Provide nameSizeFile if output results to files.");
		this->needSave = false;
	} else {
		ifstream infile(SLPROP(DetectionOutput, saveOutputParam).nameSizeFile.c_str());
		SASSERT(infile.good(),
				"Failed to open name size file: %s", SLPROP(DetectionOutput, saveOutputParam).nameSizeFile.c_str());
		// The file is in the following format:
		// 	name height width
		// 	...
		string name;
		int height;
		int width;
		while (infile >> name >> height >> width) {
			this->names.push_back(name);
			this->sizes.push_back(std::make_pair(height, width));
		}
		infile.close();
		if (SLPROP(DetectionOutput, saveOutputParam).numTestImage >= 0) {
			//this->numTestImage = SLPROP(DetectionOutput, saveOutputParam).numTestImage;
		} else {
			SLPROP(DetectionOutput, saveOutputParam).numTestImage = this->names.size();
		}
		SASSERT0(SLPROP(DetectionOutput, saveOutputParam).numTestImage <= this->names.size());
	}
	this->nameCount = 0;

	if (SLPROP(DetectionOutput, visualize)) {
		//this->visualizeThresh = builder->_visualizeThresh;
	}
}

template <typename Dtype>
DetectionOutputLayer<Dtype>::~DetectionOutputLayer() {

}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}
	if (!inputShapeChanged) return;

	this->bboxPreds.reshapeLike(this->_inputData[0]);
	if (!SLPROP(DetectionOutput, shareLocation)) {
		this->bboxPermute.reshapeLike(this->_inputData[0]);
	}
	this->confPermute.reshapeLike(this->_inputData[1]);


	if (this->needSave) {
		SASSERT0(this->nameCount <= this->names.size());
		if (this->nameCount % SLPROP(DetectionOutput, saveOutputParam).numTestImage == 0) {
			// Clean all outputs.
			if (SLPROP(DetectionOutput, saveOutputParam).outputFormat == "VOC") {
				boost::filesystem::path outputDirectory(SLPROP(DetectionOutput, saveOutputParam).outputDirectory);
				for (map<int, string>::iterator it = this->labelToName.begin();
						it != this->labelToName.end(); it++) {
					if (it->first == SLPROP(DetectionOutput, backgroundLabelId)) {
						continue;
					}
					std::ofstream outfile;
					boost::filesystem::path file(SLPROP(DetectionOutput, saveOutputParam).outputNamePrefix + it->second + ".txt");
					boost::filesystem::path _outfile = outputDirectory / file;
					outfile.open(_outfile.string().c_str(), std::ofstream::out);
				}
			}
		}
	}

	SASSERT0(this->_inputData[0]->batches() == this->_inputData[1]->batches());
	if (this->bboxPreds.batches() != this->_inputData[0]->batches() ||
			this->bboxPreds.getCountByAxis(1) != this->_inputData[0]->getCountByAxis(1)) {
		this->bboxPreds.reshapeLike(this->_inputData[0]);
	}
	if (!SLPROP(DetectionOutput, shareLocation) && (this->bboxPermute.batches() != this->_inputData[0]->batches()
			|| this->bboxPermute.getCountByAxis(1) != this->_inputData[0]->getCountByAxis(1))) {
		this->bboxPermute.reshapeLike(this->_inputData[0]);
	}
	if (this->confPermute.batches() != this->_inputData[1]->batches() ||
			this->confPermute.getCountByAxis(1) != this->_inputData[1]->getCountByAxis(1)) {
		this->confPermute.reshapeLike(this->_inputData[1]);
	}

	this->numPriors = this->_inputData[2]->channels() / 4;

	//cout << "numPriors: " << this->numPriors << ", numLocClasses: " << this->numLocClasses << endl;
	//this->_inputData[0]->print_shape();
	SASSERT(this->numPriors * this->numLocClasses * 4 == this->_inputData[0]->channels(),
			"Number of priors must match number of location predictions.");

	//cout << "numClasses: " << SLPROP(DetectionOutput, numClasses) << endl;
	//this->_inputData[1]->print_shape();
	SASSERT(this->numPriors * SLPROP(DetectionOutput, numClasses) == this->_inputData[1]->channels(),
			"Number of priors must match number of confidence predictions.");
	// num() and channels() are 1.
	vector<uint32_t> outputShape(4, 1);
	// Since the number of bboxes to be kept is unknown before nms, we manually
	// set it to (fake) 1.
	outputShape[2] = 1;
	// Each orw is a 7 dimension vector, which stores
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
	outputShape[3] = 7;
	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* locData = this->_inputData[0]->device_data();
	const Dtype* priorData = this->_inputData[2]->device_data();
	const int num = this->_inputData[0]->batches();

	// Decode predictions.
	Dtype* bboxData = this->bboxPreds.mutable_device_data();
	const int locCount = this->bboxPreds.getCount();
	const bool clipBBox = false;
	DecodeBBoxesGPU<Dtype>(locCount, locData, priorData, SLPROP(DetectionOutput, codeType),
			SLPROP(DetectionOutput, varianceEncodedInTarget), this->numPriors, SLPROP(DetectionOutput, shareLocation),
			this->numLocClasses, SLPROP(DetectionOutput, backgroundLabelId), clipBBox, bboxData);

	// Retrieve all decoded location predictions.
	const Dtype* bboxHostData;
	if (!SLPROP(DetectionOutput, shareLocation)) {
		Dtype* bboxPermuteData = this->bboxPermute.mutable_device_data();
		PermuteDataGPU<Dtype>(locCount, bboxData, this->numLocClasses, this->numPriors, 4,
				bboxPermuteData);
		bboxHostData = this->bboxPermute.host_data();
	} else {
		bboxHostData = this->bboxPreds.host_data();
	}

	// Retrieve all confidences.
	Dtype* confPermuteData = this->confPermute.mutable_device_data();
	PermuteDataGPU<Dtype>(this->_inputData[1]->getCount(), this->_inputData[1]->device_data(),
			SLPROP(DetectionOutput, numClasses), this->numPriors, 1, confPermuteData);
	const Dtype* confHostData = this->confPermute.host_data();

	int numKept = 0;
	vector<map<int, vector<int>>> allIndices;
	for (int i = 0; i < num; i++) {
		map<int, vector<int>> indices;
		int numDet = 0;
		const int confIdx = i * SLPROP(DetectionOutput, numClasses) * this->numPriors;
		int bboxIdx;
		if (SLPROP(DetectionOutput, shareLocation)) {
			bboxIdx = i * this->numPriors * 4;
		} else {
			bboxIdx = confIdx * 4;
		}
		for (int c = 0; c < SLPROP(DetectionOutput, numClasses); c++) {
			if (c == SLPROP(DetectionOutput, backgroundLabelId)) {
				// Ignore background class.
				continue;
			}
			const Dtype* curConfData = confHostData + confIdx + c * this->numPriors;
			const Dtype* curBBoxData = bboxHostData + bboxIdx;
			if (!SLPROP(DetectionOutput, shareLocation)) {
				curBBoxData += c * this->numPriors * 4;
			}
			ApplyNMSFast(curBBoxData, curConfData, this->numPriors, SLPROP(DetectionOutput, confidenceThreshold),
					SLPROP(DetectionOutput, nmsParam).nmsThreshold, SLPROP(DetectionOutput, nmsParam).eta, SLPROP(DetectionOutput, nmsParam).topK, &(indices[c]));
			numDet += indices[c].size();
		}
		if (SLPROP(DetectionOutput, keepTopK) > -1 && numDet > SLPROP(DetectionOutput, keepTopK)) {
			vector<pair<float, pair<int, int>>> scoreIndexPairs;
			for (map<int, vector<int>>::iterator it = indices.begin();
					it != indices.end(); it++) {
				int label = it->first;
				const vector<int>& labelIndices = it->second;
				for (int j = 0; j < labelIndices.size(); j++) {
					int idx = labelIndices[j];
					float score = confHostData[confIdx + label * this->numPriors + idx];
					scoreIndexPairs.push_back(
							std::make_pair(score, std::make_pair(label, idx)));
				}
			}

			// Keep top k results per image.
			std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
					SortScorePairDescend<pair<int, int>>);
			scoreIndexPairs.resize(SLPROP(DetectionOutput, keepTopK));
			// Store the new indices.
			map<int, vector<int>> newIndices;
			for (int j = 0; j < scoreIndexPairs.size(); j++) {
				int label = scoreIndexPairs[j].second.first;
				int idx = scoreIndexPairs[j].second.second;
				newIndices[label].push_back(idx);
			}
			allIndices.push_back(newIndices);
			numKept += SLPROP(DetectionOutput, keepTopK);
		} else {
			allIndices.push_back(indices);
			numKept += numDet;
		}
	}

	vector<uint32_t> outputShape(4, 1);
	outputShape[2] = numKept;
	outputShape[3] = 7;
	Dtype* outputData;
	if (numKept == 0) {
        if (SNPROP(status) == NetworkStatus::Test) {
		    STDOUT_LOG("Couldn't find any detections.");
        }
		outputShape[2] = num;
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
		soooa_set<Dtype>(this->_outputData[0]->getCount(), -1, outputData);
		// Generate fake results per image.
		for (int i = 0; i < num; i++) {
			outputData[0] = i;
			outputData += 7;
		}
	} else {
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
	}

	int count = 0;
	boost::filesystem::path outputDirectory(SLPROP(DetectionOutput, saveOutputParam).outputDirectory);
	for (int i = 0; i < num; i++) {
		const int confIdx = i * SLPROP(DetectionOutput, numClasses) * this->numPriors;
		int bboxIdx;
		if (SLPROP(DetectionOutput, shareLocation)) {
			bboxIdx = i * this->numPriors * 4;
		} else {
			bboxIdx = confIdx * 4;
		}
		for (map<int, vector<int>>::iterator it = allIndices[i].begin();
				it != allIndices[i].end(); it++) {
			int label = it->first;
			vector<int>& indices = it->second;
			if (this->needSave) {
				SASSERT(this->labelToName.find(label) != this->labelToName.end(),
						"Cannot find label: %d in the label map.", label);
				SASSERT0(this->nameCount < this->names.size());
			}
			const Dtype* curConfData = confHostData + confIdx + label * this->numPriors;
			const Dtype* curBBoxData = bboxHostData + bboxIdx;
			if (!SLPROP(DetectionOutput, shareLocation)) {
				curBBoxData += label * this->numPriors * 4;
			}
			for (int j = 0; j < indices.size(); j++) {
				int idx = indices[j];
				outputData[count * 7] = i;
				outputData[count * 7 + 1] = label;
				outputData[count * 7 + 2] = curConfData[idx];
				for (int k = 0; k < 4; k++) {
					outputData[count * 7 + 3 + k] = curBBoxData[idx * 4 + k];
				}
				if (this->needSave) {
					// Generate output bbox.
					NormalizedBBox bbox;
					bbox.xmin = outputData[count * 7 + 3];
					bbox.ymin = outputData[count * 7 + 4];
					bbox.xmax = outputData[count * 7 + 5];
					bbox.ymax = outputData[count * 7 + 6];
					NormalizedBBox outBBox;
					OutputBBox(bbox, this->sizes[this->nameCount], false, &outBBox);
					float score = outputData[count * 7 + 2];
					float xmin = outBBox.xmin;
					float ymin = outBBox.ymin;
					float xmax = outBBox.xmax;
					float ymax = outBBox.ymax;

					ptree ptXmin;
					ptree ptYmin;
					ptree ptWidth;
					ptree ptHeight;
					ptXmin.put<float>("", round(xmin * 100) / 100.);
					ptYmin.put<float>("", round(ymin * 100) / 100.);
					ptWidth.put<float>("", round((xmax - xmin) * 100) / 100.);
					ptHeight.put<float>("", round((ymax - ymin) * 100) / 100.);

					ptree curBBox;
					curBBox.push_back(std::make_pair("", ptXmin));
					curBBox.push_back(std::make_pair("", ptYmin));
					curBBox.push_back(std::make_pair("", ptWidth));
					curBBox.push_back(std::make_pair("", ptHeight));

					ptree curDet;
					curDet.put("image_id", this->names[this->nameCount]);
					if (SLPROP(DetectionOutput, saveOutputParam).outputFormat == "ILSVRC") {
						curDet.put<int>("category_id", label);
					} else {
						curDet.put("category_id", this->labelToName[label].c_str());
					}
					curDet.add_child("bbox", curBBox);
					curDet.put<float>("score", score);

					this->detections.push_back(std::make_pair("", curDet));
				}
				count++;
			}
		}

		if (this->needSave) {
			this->nameCount++;
			//cout << "nameCount: " << this->nameCount << ", numTestImage: " << SLPROP(DetectionOutput, saveOutputParam).numTestImage << endl;
			if (this->nameCount % SLPROP(DetectionOutput, saveOutputParam).numTestImage == 0) {
				cout << "meet the condition!" << endl;
				if (SLPROP(DetectionOutput, saveOutputParam).outputFormat == "VOC") {
					map<string, std::ofstream*> outfiles;
					for (int c = 0; c < SLPROP(DetectionOutput, numClasses); c++) {
						if (c == SLPROP(DetectionOutput, backgroundLabelId)) {
							continue;
						}
						string labelName = this->labelToName[c];
						boost::filesystem::path file(
								SLPROP(DetectionOutput, saveOutputParam).outputNamePrefix + labelName + ".txt");
						boost::filesystem::path outfile = outputDirectory / file;
						outfiles[labelName] = NULL;
						SNEW(outfiles[labelName], std::ofstream, outfile.string().c_str(),
								std::ofstream::out);
						SASSUME0(outfiles[labelName] != NULL);
					}
					BOOST_FOREACH(ptree::value_type& det, this->detections.get_child("")) {
						ptree pt = det.second;
						string labelName = pt.get<string>("category_id");
						if (outfiles.find(labelName) == outfiles.end()) {
							std::cout << "Cannot find " << labelName << endl;
							continue;
						}
						string imageName = pt.get<string>("image_id");
						float score = pt.get<float>("score");
						vector<int> bbox;
						BOOST_FOREACH(ptree::value_type& elem, pt.get_child("bbox")) {
							bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
						}
						*(outfiles[labelName]) << imageName;
						*(outfiles[labelName]) << " " << score;
						*(outfiles[labelName]) << " " << bbox[0] << " " << bbox[1];
						*(outfiles[labelName]) << " " << bbox[0] + bbox[2];
						*(outfiles[labelName]) << " " << bbox[1] + bbox[3];
						*(outfiles[labelName]) << endl;
					}
					for (int c = 0; c < SLPROP(DetectionOutput, numClasses); c++) {
						if (c == SLPROP(DetectionOutput, backgroundLabelId)) {
							continue;
						}
						string labelName = this->labelToName[c];
						outfiles[labelName]->flush();
						outfiles[labelName]->close();
						SDELETE(outfiles[labelName]);
					}
				} else if (SLPROP(DetectionOutput, saveOutputParam).outputFormat == "COCO") {
					SASSERT(false, "COCO is not supported");
				} else if (SLPROP(DetectionOutput, saveOutputParam).outputFormat == "ILSVRC") {
					SASSERT(false, "ILSVRC is not supported");
				}
				this->nameCount = 0;
				this->detections.clear();
			}
		}
	}
	if (SLPROP(DetectionOutput, visualize)) {
#if 0
		vector<cv::Mat> cvImgs;
		const int singleImageSize = this->_inputData[3]->getCountByAxis(1);
		const int imageHeight = 300;
		const int imageWidth = 300;
		const int height = 0;
		const int width = 0;
		const vector<Dtype> pixelMeans = {};
		const Dtype* dataData = this->_inputData[3]->host_data();
		transformInv(num, singleImageSize,
				imageHeight, imageWidth,
				height, width, pixelMeans,
				dataData, this->temp);
		vector<cv::Scalar> colors = GetColors(this->labelToDisplayName.size());
		VisualizeBBox(cvImgs, this->_outputData[0], SLPROP(DetectionOutput, visualizeThresh), colors,
				this->labelToDisplayName, this->saveFile);
#endif
	}
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::backpropagation() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DetectionOutputLayer<Dtype>::initLayer() {
	DetectionOutputLayer* layer = NULL;
	SNEW(layer, DetectionOutputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void DetectionOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    DetectionOutputLayer<Dtype>* layer = (DetectionOutputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void DetectionOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 4);
	} else {
		SASSERT0(index < 1);
	}

    DetectionOutputLayer<Dtype>* layer = (DetectionOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DetectionOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DetectionOutputLayer<Dtype>* layer = (DetectionOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DetectionOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DetectionOutputLayer<Dtype>* layer = (DetectionOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DetectionOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	DetectionOutputLayer<Dtype>* layer = (DetectionOutputLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void DetectionOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool DetectionOutputLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 3)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 7;
    outputShape.push_back(outputShape1);

    return true;
}

template<typename Dtype>
uint64_t DetectionOutputLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {

    size_t size = 0;
    const int input0Count = tensorCount(inputShape[0]);
    const int input1Count = tensorCount(inputShape[1]);
    
    // bboxPreds
    size += ALIGNUP(sizeof(Dtype) * input0Count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    // bboxPermute
	if (!SLPROP(DetectionOutput, shareLocation)) {
        size += ALIGNUP(sizeof(Dtype) * input0Count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;
	}
    
    // confPermute
    size += ALIGNUP(sizeof(Dtype) * input1Count, SPARAM(CUDA_MEMPAGE_SIZE)) * 2UL;

    return size;
}

template class DetectionOutputLayer<float>;



































