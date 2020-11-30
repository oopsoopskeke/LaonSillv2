/*
 * ProposalLayer.cpp
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#include <vector>


#include "ProposalLayer.h"
#include "GenerateAnchorsUtil.h"
#include "BboxTransformUtil.h"
#include "frcnn_common.h"
#include "Network.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"


#define PROPOSALLAYER_LOG 0


using namespace std;


template <typename Dtype>
ProposalLayer<Dtype>::ProposalLayer()
	: Layer<Dtype>() {
	this->type = Layer<Dtype>::Proposal;

	GenerateAnchorsUtil::generateAnchors(this->anchors, SLPROP(Proposal, scales));
	this->numAnchors = this->anchors.size();

#if PROPOSALLAYER_LOG
	cout << "featStride: " << SLPROP(Proposal, featStride) << endl;
	print2dArray("anchors", this->anchors);
#endif
}

template <typename Dtype>
ProposalLayer<Dtype>::~ProposalLayer() {

}

template <typename Dtype>
void ProposalLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// rois blob: holds R regions of interest, each is a 5-tuple
		// (n, x1, y1, x2, y2) specifying an image batch index n and a
		// rectangle (x1, y1, x2, y2)
		this->_outputData[0]->reshape({1, 1, 1, 5});

		// scores data: holds scores for R regions of interest
		if (this->_outputData.size() > 1) {
			this->_outputData[1]->reshape({1, 1, 1, 1});
		}
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::feedforward() {
	reshape();

	// Algorithm:
	//
	// for each (H, W) location i
	//		generate A anchor boxes centered on cell i
	//		apply predicted bbox deltas at cell i to each of the A 
    //		ancscoreData->getShape()hors
	// clip predicted boxes to image
	// remove predicted boxes with either height or width < threshold
	// sort all (proposal, score) pairs by score from hightest to lowest
	// take top pre_nms_topN proposals before NMS
	// apply NMS with threshold 0.7 to remaining proposals
	// take after_nms_topN proposals after NMS
	// return the top proposals (-> RoIs top, scores top)

	assert(this->_inputData[0]->getShape(0) == 1 &&
			"Only single item batches are supported");


	uint32_t preNmsTopN;
	uint32_t postNmsTopN;
	float nmsThresh;
	uint32_t minSize;

	if (SNPROP(status) == NetworkStatus::Train) {
		preNmsTopN 	= TRAIN_RPN_PRE_NMS_TOP_N;
		postNmsTopN	= TRAIN_RPN_POST_NMS_TOP_N;
		nmsThresh 	= TRAIN_RPN_NMS_THRESH;
		minSize 	= TRAIN_RPN_MIN_SIZE;
	} else if (SNPROP(status) == NetworkStatus::Test) {
		preNmsTopN 	= TEST_RPN_PRE_NMS_TOP_N;
		postNmsTopN = TEST_RPN_POST_NMS_TOP_N;
		nmsThresh 	= TEST_RPN_NMS_THRESH;
		minSize 	= TEST_RPN_MIN_SIZE;
	}


	// the first set of numAnchors channels are bg probs
	// the second set are the fg probs, which we want
	//Data<Dtype>* scoresData = NULL;
	//SNEW(scoresData, Data<Dtype>, "scoresData");
	//SASSUME0(scoresData != NULL);
    std::shared_ptr<Data<Dtype>> scoresData = std::make_shared<Data<Dtype>>("scoresData");
	this->_inputData[0]->range({0, (int)this->numAnchors, 0, 0}, {-1, -1, -1, -1}, scoresData.get());


#if PROPOSALLAYER_LOG
	this->_printOn();
	this->_inputData[1]->print_data({}, false);
	scoresData->print_data({}, false);
	this->_printOff();
#endif

	//Data<Dtype>* bboxDeltas = NULL;
	//SNEW(bboxDeltas, Data<Dtype>, "bboxDeltas");
	//SASSUME0(bboxDeltas != NULL);
    std::shared_ptr<Data<Dtype>> bboxDeltas = std::make_shared<Data<Dtype>>("bboxDeltas");
	bboxDeltas->reshapeLike(this->_inputData[1]);
	bboxDeltas->set_host_data(this->_inputData[1]);
    
	//Data<Dtype>* imInfo = NULL;
	//SNEW(imInfo, Data<Dtype>, "imInfo");
	//SASSUME0(imInfo != NULL);
    std::shared_ptr<Data<Dtype>> imInfo = std::make_shared<Data<Dtype>>("imInfo");
	this->_inputData[2]->range({0, 0, 0, 0}, {-1, -1, 1, -1}, imInfo.get());

#if PROPOSALLAYER_LOG
	cout << "im_size: (" << imInfo->host_data()[0] << ", " <<
			imInfo->host_data()[1] << ")" << endl;
	cout << "scale: " << imInfo->host_data()[2] << endl;
	this->_printOn();
	bboxDeltas->print_data({}, false);
	imInfo->print_data({}, false);
	this->_printOff();
#endif

	// 1. Generate propsals from bbox deltas and shifted anchors
	const uint32_t height = scoresData->getShape(2);
	const uint32_t width = scoresData->getShape(3);

#if PROPOSALLAYER_LOG
	cout << "score map size: " << scoresData->getShape(0) << ", " <<
        scoresData->getShape(1) << ", " << scoresData->getShape(2) << ", " <<
        scoresData->getShape(3) << endl;
#endif

	// Enumerate all shifts
	const uint32_t numShifts = height * width;
	vector<vector<uint32_t>> shifts(numShifts);

	for (uint32_t i = 0; i < numShifts; i++)
		shifts[i].resize(4);

	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			vector<uint32_t>& shift = shifts[i*width+j];
			shift[0] = j * SLPROP(Proposal, featStride);
			shift[2] = j * SLPROP(Proposal, featStride);
			shift[1] = i * SLPROP(Proposal, featStride);
			shift[3] = i * SLPROP(Proposal, featStride);
		}
	}
#if PROPOSALLAYER_LOG
	print2dArray("shifts", shifts);
#endif

	// Enumerate all shifted anchors:
	//
	// add A anchors (1, A, 4) to
	// cell K shifts (K, 1, 4) to get
	// shift anchors (K, A, 4)
	// reshape to (K*A, 4) shifted anchors
	const uint32_t A = this->numAnchors;
	const uint32_t K = shifts.size();
	const uint32_t totalAnchors = K * A;

	vector<vector<float>> anchors(totalAnchors);
	uint32_t anchorIndex;
	for (uint32_t i = 0; i < K; i++) {
		for (uint32_t j = 0; j < A; j++) {
			anchorIndex = i * A + j;
			vector<float>& anchor = anchors[anchorIndex];
			anchor.resize(4);
			anchor[0] = this->anchors[j][0] + shifts[i][0];
			anchor[1] = this->anchors[j][1] + shifts[i][1];
			anchor[2] = this->anchors[j][2] + shifts[i][2];
			anchor[3] = this->anchors[j][3] + shifts[i][3];
		}
	}
#if PROPOSALLAYER_LOG
	print2dArray("anchors", anchors);
#endif

	// Transpose and reshape predicted bbox transformations to get them
	// into the same order as the anchors:
	//
	// bbox deltas will be (1, 4 * A, H, W) format
	// transpose to (1, H, W, 4 * A)
	// reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
	// in slowest to fastest order

	bboxDeltas->transpose({0, 2, 3, 1});
	bboxDeltas->reshapeInfer({1, 1, -1, 4});

	// Same stroy for the scores:
	//
	// scores are (1, A, H, W) format
	// transpose to (1, H, W, A)
	// reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
	scoresData->transpose({0, 2, 3, 1});
	//scoresData->reshapeInfer({1, 1, -1, 1});
	// XXX: 정렬에 용이할 것 같아서 바꿔 봄.
	scoresData->reshapeInfer({1, 1, 1, -1});
	vector<float> scores;
	fill1dVecWithData(scoresData.get(), scores);
	//SDELETE(scoresData);

#if PROPOSALLAYER_LOG
	this->_printOn();
	bboxDeltas->print_data({}, false);
	this->_printOff();
#endif
	// Convert anchors into proposals via bbox transformations
	vector<vector<Dtype>> proposals;
	BboxTransformUtil::bboxTransformInv(anchors, bboxDeltas.get(), proposals);
	//SDELETE(bboxDeltas);

#if PROPOSALLAYER_LOG
	printArray("scores", scores);
	print2dArray("proposals", proposals);
#endif

	// 2. clip predicted boxes to image
	BboxTransformUtil::clipBoxes(proposals,
			{imInfo->host_data()[0], imInfo->host_data()[1]});
#if PROPOSALLAYER_LOG
	cout << imInfo->host_data()[0] << "x" << imInfo->host_data()[1] << endl;
	print2dArray("proposals", proposals);
#endif

	// 3. remove predicted boxes with either height or width < threshold
	// (NOTE: convert minSize to input image scale stored in imInfo[2])
	vector<uint32_t> keep;
	_filterBoxes(proposals, minSize * imInfo->host_data()[2], keep);

	//cout << "num of proposals: " << proposals.size() << ", keep: " << keep.size() << endl;

	proposals = vec_keep_by_index(proposals, keep);
	scores = vec_keep_by_index(scores, keep);
#if PROPOSALLAYER_LOG
	printArray("keep", keep);
	print2dArray("proposals", proposals);
	printArray("scores", scores);
#endif

	// 4. sort all (proposal, score) pairs by score from highest to lowest
	// 5. take preNmsTopN (e.g. 6000)
	vector<uint32_t> order(scores.size());
#if !SOOOA_DEBUG
	iota(order.begin(), order.end(), 0);
	vec_argsort(scores, order);
	/*
	std::cout << "proposal layer sort result: " << std::endl;
	for (int i = 0; i < scores.size(); i++) {
		std::cout << "\tscore: " << scores[order[i]] << std::endl;
	}
	*/
#else
	//string path = "/home/jkim/Dev/data/numpy_array/order.npz";
	//loadPredefinedOrder(path, order);
	iota(order.begin(), order.end(), 0);
	vec_argsort(scores, order);
#endif

#if PROPOSALLAYER_LOG
	for (uint32_t i = 0; i < order.size(); i++) {
		cout << order[i] << "\t: " << scores[order[i]] << endl;
	}
#endif
	if (preNmsTopN > 0 && preNmsTopN < order.size())
		order.erase(order.begin() + preNmsTopN, order.end());
	proposals = vec_keep_by_index(proposals, order);
	scores = vec_keep_by_index(scores, order);

#if PROPOSALLAYER_LOG
	printArray("order", order);
	print2dArray("proposals", proposals);
	printArray("scores", scores);
#endif

	// 6. apply nms (e.g. threshold = 0.7)
	// 7. take postNmsTopN (e.g. 300)
	// 8. return the top proposals (->RoIs top)

	nms(proposals, scores, nmsThresh, keep);

	if (postNmsTopN > 0 && postNmsTopN < keep.size())
		keep.erase(keep.begin() + postNmsTopN, keep.end());
	proposals = vec_keep_by_index(proposals, keep);
	scores = vec_keep_by_index(scores, keep);

#if PROPOSALLAYER_LOG
	printArray("keep", keep);
	print2dArray("proposals", proposals);
	printArray("scores", scores);
#endif

	// Output rois data
	// Our RPN implementation only supports a single input image, so all
	// batch inds are 0
	vec_2d_pad(1, proposals);
	this->_outputData[0]->reshape({1, 1, (uint32_t)proposals.size(), 5});
	this->_outputData[0]->fill_host_with_2d_vec(proposals, {0, 1, 2, 3});

#if PROPOSALLAYER_LOG
	cout << "# of proposals: " << proposals.size() << endl;
	print2dArray("proposals", proposals);
	this->_printOn();
	this->_outputData[0]->print_data({}, false);
	this->_printOff();
	printArray("scores", scores);
#endif

	// XXX:
	// [Optional] output scores data
	if (this->_outputData.size() > 1) {
		assert(this->_outputData.size() == 1);
	}
	//SDELETE(imInfo);


	/*
	cout << "rois shape: " << endl;
	print2dArray("rois", proposals);

	const string windowName = "rois";
	uint32_t numBoxes = proposals.size();

	Dtype scale = this->_inputData[2]->host_data()[2];
	int boxOffset = 1;
	cout << "scale: " << scale << endl;
	const int onceSize = 5;

	for (int j = 0; j < (numBoxes / onceSize); j++) {
		cv::Mat im = cv::imread(Util::imagePath, CV_LOAD_IMAGE_COLOR);
		cv::resize(im, im, cv::Size(), scale, scale, CV_INTER_LINEAR);

		for (uint32_t i = j*onceSize; i < (j+1)*onceSize; i++) {
			cv::rectangle(im, cv::Point(proposals[i][boxOffset+0], proposals[i][boxOffset+1]),
				cv::Point(proposals[i][boxOffset+2], proposals[i][boxOffset+3]),
				cv::Scalar(0, 0, 255), 2);
		}

		cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName, im);

		if (pause) {
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}
	*/
}

template <typename Dtype>
void ProposalLayer<Dtype>::backpropagation() {

}


template <typename Dtype>
void ProposalLayer<Dtype>::_filterBoxes(std::vector<std::vector<float>>& boxes,
		const float minSize, std::vector<uint32_t>& keep) {
	// Remove all boxes with any side smaller than minSize
	keep.clear();
	float ws, hs;
	const uint32_t numBoxes = boxes.size();
	for (uint32_t i = 0; i < numBoxes; i++) {
		std::vector<float>& box = boxes[i];
		ws = box[2] - box[0] + 1;
		hs = box[3] - box[1] + 1;

		if (ws >= minSize && hs >= minSize)
			keep.push_back(i);
		//else
		//	cout << "ws: " << ws << ", hs:" << hs << endl;
	}
	//exit(1);
}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ProposalLayer<Dtype>::initLayer() {
	ProposalLayer* layer = NULL;
	SNEW(layer, ProposalLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ProposalLayer<Dtype>::destroyLayer(void* instancePtr) {
    ProposalLayer<Dtype>* layer = (ProposalLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ProposalLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 3);
	} else {
		SASSERT0(index < 1);
	}

    ProposalLayer<Dtype>* layer = (ProposalLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ProposalLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ProposalLayer<Dtype>* layer = (ProposalLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ProposalLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ProposalLayer<Dtype>* layer = (ProposalLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void ProposalLayer<Dtype>::backwardTensor(void* instancePtr) {
	ProposalLayer<Dtype>* layer = (ProposalLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void ProposalLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ProposalLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 3)
        return false;

    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 5;
    outputShape.push_back(outputShape1);

    TensorShape outputShape2;
    outputShape2.N = 1;
    outputShape2.C = 1;
    outputShape2.H = 1;
    outputShape2.W = 1;
    outputShape.push_back(outputShape2);

    return true;
}

template<typename Dtype>
uint64_t ProposalLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ProposalLayer<float>;
