/*
 * ProposalTargetLayer.cpp
 *
 *  Created on: Nov 30, 2016
 *      Author: jkim
 */


#include "ProposalTargetLayer.h"
#include "frcnn_common.h"
#include "BboxTransformUtil.h"
#include "RoIDBUtil.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

#define PROPOSALTARGETLAYER_LOG 0

using namespace std;


template <typename Dtype>
ProposalTargetLayer<Dtype>::ProposalTargetLayer()
	: Layer<Dtype>() {
	this->type = Layer<Dtype>::ProposalTarget;
}

template <typename Dtype>
ProposalTargetLayer<Dtype>::~ProposalTargetLayer() {}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// sampled rois (0, x1, y1, x2, y2)
		this->_outputData[0]->reshape({1, 1, 1, 5});
		// labels
		this->_outputData[1]->reshape({1, 1, 1, 1});
		// bbox_targets
		this->_outputData[2]->reshape({1, 1, 1, SLPROP(ProposalTarget, numClasses) * 4});
		// bbox_inside_weights
		this->_outputData[3]->reshape({1, 1, 1, SLPROP(ProposalTarget, numClasses) * 4});
		// bbox_outside_weights
		this->_outputData[4]->reshape({1, 1, 1, SLPROP(ProposalTarget, numClasses) * 4});
	}
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::feedforward() {
	reshape();

	// Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
	// (i,e., ProposalLayer), or any other source
	vector<vector<float>> allRois;
	fill2dVecWithData(this->_inputData[0], allRois);

	// GT boxes (x1, y1, x2, y2, label)
	// TODO(rbg): it's annoying that sometimes I have extra info before
	// and other times after box coordinates -- normalize to one format
	vector<vector<float>> gtBoxes;
	fill2dVecWithData(this->_inputData[1], gtBoxes);

#if PROPOSALTARGETLAYER_LOG
	cout << "# of all rois: " << allRois.size() << endl;
	print2dArray("allRois", allRois);
	print2dArray("gtBoxes", gtBoxes);
#endif

	// Include ground-truth boxes in the set of candidate rois
	for (uint32_t i = 0; i < gtBoxes.size(); i++) {
		vector<float> gtBox(5);
		gtBox[1] = gtBoxes[i][0];
		gtBox[2] = gtBoxes[i][1];
		gtBox[3] = gtBoxes[i][2];
		gtBox[4] = gtBoxes[i][3];
		allRois.push_back(gtBox);
	}
#if PROPOSALTARGETLAYER_LOG
	print2dArray("allRois", allRois);
#endif

	// Sanity check: single batch only
	// assert(np.all(allRois ...

	const uint32_t numImages = 1;
	const uint32_t roisPerImage = TRAIN_BATCH_SIZE / numImages;
	const uint32_t fgRoisPerImage = np_round(TRAIN_FG_FRACTION * roisPerImage);

	// Sample rois with classification labels and bounding box regression
	// targets
	vector<uint32_t> labels;
	vector<vector<float>> rois;
	vector<vector<float>> bboxTargets;
	vector<vector<float>> bboxInsideWeights;
	_sampleRois(allRois, gtBoxes, fgRoisPerImage, roisPerImage,
			labels, rois, bboxTargets, bboxInsideWeights);

	// sampled rois
	this->_outputData[0]->reshape({1, 1, (uint32_t)rois.size(), (uint32_t)rois[0].size()});
	this->_outputData[0]->fill_host_with_2d_vec(rois);



	// classification labels
	this->_outputData[1]->reshape({1, 1, 1, (uint32_t)labels.size()});
	this->_outputData[1]->fill_host_with_1d_vec(labels);

	const uint32_t numTargets = bboxTargets.size();
	const uint32_t numTargetElem = bboxTargets[0].size();

	// bboxTargets
	this->_outputData[2]->reshape({1, 1, numTargets, numTargetElem});
	this->_outputData[2]->fill_host_with_2d_vec(bboxTargets);
	// bboxTargets가 사용될 Fully Connected Layer에 맞춰서 shape 조정
	//this->_outputData[2]->reshape({numTargets, 1, numTargetElem, 1});

#if PROPOSALTARGETLAYER_LOG
	cout << "# of rois: " << rois.size() << endl;
	cout << "# of targets: " << numTargets << endl;
#endif

	/*
	Data<Dtype>::printConfig = true;
	this->_outputData[0]->print_data({}, false);
	this->_outputData[1]->print_data({}, false);
	this->_outputData[2]->print_data({}, false);
	Data<Dtype>::printConfig = false;
	exit(1);
	*/

	// bboxInsideWeights
	this->_outputData[3]->reshape({1, 1, numTargets, numTargetElem});
	this->_outputData[3]->fill_host_with_2d_vec(bboxInsideWeights);
	//this->_outputData[3]->reshape({numTargets, 1, numTargetElem, 1});

	// bboxOutsideWeights
	vector<vector<float>> bboxOutsideWeights(numTargets);
	for (uint32_t i = 0; i < numTargets; i++) {
		bboxOutsideWeights[i].resize(numTargetElem);
		for (uint32_t j = 0; j < numTargetElem; j++) {
			if (bboxInsideWeights[i][j] > 0.0f)
				bboxOutsideWeights[i][j] = 1.0f;
		}
	}
	this->_outputData[4]->reshape({1, 1, numTargets, numTargetElem});
	this->_outputData[4]->fill_host_with_2d_vec(bboxOutsideWeights);
	//this->_outputData[4]->reshape({numTargets, 1, numTargetElem, 1});

	/*
	Data<Dtype>::printConfig = true;
	this->_outputData[0]->print();
	this->_outputData[2]->print();
	Data<Dtype>::printConfig = false;

	if (this->_outputData[0]->getShape(2) < 128) {
		cout << endl;
	}
	*/

	assert(this->_outputData[0]->getShape(2) == this->_outputData[2]->getShape(2));
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::backpropagation() {
	// This layer does not propagate gradients.
}



template <typename Dtype>
void ProposalTargetLayer<Dtype>::_sampleRois(
		const vector<vector<float>>& allRois,
		const vector<vector<float>>& gtBoxes,
		const uint32_t fgRoisPerImage,
		const uint32_t roisPerImage,
		vector<uint32_t>& labels,
		vector<vector<float>>& rois,
		vector<vector<float>>& bboxTargets,
		vector<vector<float>>& bboxInsideWeights) {

	// Generate a random sample of RoIs comprising foreground and background
	// examples.

	// overlaps: (rois x gtBoxes)
	vector<vector<float>> overlaps;		// 각 ex-roi와 각 gt-roi별 IoU 값
	RoIDBUtil::bboxOverlaps(allRois, 1, gtBoxes, 0, overlaps);
#if PROPOSALTARGETLAYER_LOG
	print2dArray("overlaps", overlaps);
#endif

	vector<uint32_t> gtAssignment;		// 각 ex-roi의 최대 일치 gt-roi index
	np_argmax(overlaps, 1, gtAssignment);
#if PROPOSALTARGETLAYER_LOG
	printArray("gtAssignment", gtAssignment);
#endif

	vector<float> maxOverlaps;			// 각 ex-roi의 최대 IoU 값
	np_maxByAxis(overlaps, maxOverlaps);
#if PROPOSALTARGETLAYER_LOG
	printArray("maxOverlaps", maxOverlaps);
#endif

	//vector<float> labels(gtBoxes.size());
	labels.resize(gtAssignment.size());
	for (uint32_t i = 0; i < gtAssignment.size(); i++)
		labels[i] = uint32_t(gtBoxes[gtAssignment[i]][4]);
#if PROPOSALTARGETLAYER_LOG
	printArray("labels", labels);
#endif

	// Select foreground RoIs as those with >= FG_THRESH overlap
	vector<uint32_t> fgInds;
	np_where_s(maxOverlaps, GE, TRAIN_FG_THRESH, fgInds);
#if PROPOSALTARGETLAYER_LOG
	printArray("fgInds", fgInds);
#endif

	// Guard against the case when an image has fewer than fgRoisPerImage
	// foreground RoIs
	const uint32_t fgRoisPerThisImage = uint32_t(std::min((int)fgRoisPerImage, (int)fgInds.size()));

	// Sample foreground regions without replacement
	if (fgInds.size() > 0) {
#if !SOOOA_DEBUG
		vector<uint32_t> tempFgInds;
		npr_choice(fgInds, fgRoisPerThisImage, tempFgInds);
		/*
		assert(tempFgInds.size() == fgRoisPerThisImage);
		cout << "proposal target layer fg inds of size: " << fgRoisPerThisImage << endl;
		for (int i = 0; i < tempFgInds.size(); i++) {
			assert(std::find(fgInds.begin(), fgInds.end(), tempFgInds[i]) != fgInds.end());
		}
		*/
		fgInds = tempFgInds;
#else
		if (fgInds.size() > fgRoisPerThisImage)
			fgInds.erase(fgInds.begin()+fgRoisPerThisImage, fgInds.end());
#endif
	}
#if PROPOSALTARGETLAYER_LOG
	printArray("fgInds", fgInds);
#endif

	// Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
	vector<uint32_t> bgInds;
	np_where(maxOverlaps, {LT, GE}, {TRAIN_BG_THRESH_HI, TRAIN_BG_THRESH_LO}, bgInds);
#if PROPOSALTARGETLAYER_LOG
	printArray("bgInds", bgInds);
#endif

	// Compute number of background RoIs to take from this image (guarding
	// against there being fewer than desired)
	const uint32_t bgRoisPerThisImage =
        uint32_t(std::min((int)(roisPerImage - fgRoisPerThisImage), (int)bgInds.size()));
	// Sample background regions without replacement
	if (bgInds.size() > 0) {
#if !SOOOA_DEBUG
		vector<uint32_t> tempBgInds;
		npr_choice(bgInds, bgRoisPerThisImage, tempBgInds);
		/*
		assert(tempBgInds.size() == bgRoisPerThisImage);
		cout << "proposal target layer bg inds of size: " << bgRoisPerThisImage << endl;
		for (int i = 0; i < tempBgInds.size(); i++) {
			assert(std::find(bgInds.begin(), bgInds.end(), tempBgInds[i]) != bgInds.end());
		}
		*/
		bgInds = tempBgInds;
#else
		if (bgInds.size() > bgRoisPerThisImage)
			bgInds.erase(bgInds.begin()+bgRoisPerThisImage, bgInds.end());
#endif
	}
#if PROPOSALTARGETLAYER_LOG
	printArray("bgInds", bgInds);
#endif


	// The indices that we're selecting (both fg and bg)
	vector<uint32_t> keepInds;
	keepInds.insert(keepInds.end(), fgInds.begin(), fgInds.end());
	keepInds.insert(keepInds.end(), bgInds.begin(), bgInds.end());
	// XXX: 디버깅 임시
	//for (uint32_t i = 0; i < roisPerImage; i++)
	//	keepInds.push_back(i);

	// Select sampled values from various arrays:
	labels = vec_keep_by_index(labels, keepInds);
#if PROPOSALTARGETLAYER_LOG
	printArray("keepInds", keepInds);
	printArray("labels", labels);
#endif

	// Clamp labels for the background RoIs to 0
	for (uint32_t i = fgRoisPerThisImage; i < labels.size(); i++)
		labels[i] = 0;
	//vector<vector<float>> rois = vec_keep_by_index(allRois, keepInds);
	rois = vec_keep_by_index(allRois, keepInds);
#if PROPOSALTARGETLAYER_LOG
	print2dArray("rois", rois);
#endif

	vector<vector<float>> gtRois(keepInds.size());
	for (uint32_t i = 0; i < keepInds.size(); i++)
		gtRois[i] = gtBoxes[gtAssignment[keepInds[i]]];
#if PROPOSALTARGETLAYER_LOG
	print2dArray("gtRois", gtRois);
#endif

	vector<vector<float>> bboxTargetData;
	_computeTargets(rois, 1, gtRois, 0, labels, bboxTargetData, 1);
#if PROPOSALTARGETLAYER_LOG


	/*
	vector<vector<float>> tempRois(1);
	for (uint32_t i = 0; i < fgRoisPerThisImage; i++) {
		tempRois[0].resize(4);
		tempRois[0][0] = rois[i][1];
		tempRois[0][1] = rois[i][2];
		tempRois[0][2] = rois[i][3];
		tempRois[0][3] = rois[i][4];

		displayBoxesOnImage(Util::imagePath, 1.4880952835f, tempRois);
	}
	*/


	print2dArray("bboxTargetData", bboxTargetData);
#endif

	_getBboxRegressionLabels(bboxTargetData, bboxTargets, bboxInsideWeights);
#if PROPOSALTARGETLAYER_LOG
	print2dArray("bboxTargets", bboxTargets);
	print2dArray("bboxInsideWeights", bboxInsideWeights);
#endif

}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::_computeTargets(
		const vector<vector<float>>& exRois,
		const uint32_t exRoisOffset,
		const vector<vector<float>>& gtRois,
		const uint32_t gtRoisOffset,
		const vector<uint32_t>& labels,
		vector<vector<float>>& targets,
		const uint32_t targetsOffset) {

	assert(exRois.size() == gtRois.size());
	//assert(exRois[0].size() == 4);
	//assert(gtRois[0].size() == 4);

	const uint32_t numRois = exRois.size();
	targets.resize(numRois);
	for (uint32_t i = 0; i < numRois; i++) {
		targets[i].resize(5);
	}
	BboxTransformUtil::bboxTransform(exRois, exRoisOffset, gtRois, gtRoisOffset,
			targets, targetsOffset);

	if (TRAIN_BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
		for (uint32_t i = 0; i < numRois; i++) {
			vector<float>& target = targets[i];
			target[targetsOffset+0] = (target[targetsOffset+0] -
					TRAIN_BBOX_NORMALIZE_MEANS[0]) / TRAIN_BBOX_NORMALIZE_STDS[0];
			target[targetsOffset+1] = (target[targetsOffset+1] -
					TRAIN_BBOX_NORMALIZE_MEANS[1]) / TRAIN_BBOX_NORMALIZE_STDS[1];
			target[targetsOffset+2] = (target[targetsOffset+2] -
					TRAIN_BBOX_NORMALIZE_MEANS[2]) / TRAIN_BBOX_NORMALIZE_STDS[2];
			target[targetsOffset+3] = (target[targetsOffset+3] -
					TRAIN_BBOX_NORMALIZE_MEANS[3]) / TRAIN_BBOX_NORMALIZE_STDS[3];
		}
	}

	//vec_2d_pad(1, targets);
	for (uint32_t i = 0; i < numRois; i++) {
		targets[i][0] = labels[i];
	}
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::_getBboxRegressionLabels(
		const vector<vector<float>>& bboxTargetsData,
		vector<vector<float>>& bboxTargets,
		vector<vector<float>>& bboxInsideWeights) {

	// Bounding-box regression targets (bboxTargetData) are stored in a
	// compact form N x (class, tx, ty, tw, th)
	//
	// This function expands thoes targets into the 4-of-4*K representation used
	// by the network (i.e. only one class has non-zero targets).
	//
	// Returns:
	//		bboxTargets: N x 4K data of regression targets
	//		bboxInsideWeights: N x 4K data of loss weights

	const uint32_t numBboxTargets = bboxTargetsData.size();
	bboxTargets.resize(numBboxTargets);
	bboxInsideWeights.resize(numBboxTargets);


	uint32_t cls;
	uint32_t start, end;
	for (uint32_t i = 0; i < numBboxTargets; i++) {
		bboxTargets[i].resize(4 * SLPROP(ProposalTarget, numClasses));
		bboxInsideWeights[i].resize(4 * SLPROP(ProposalTarget, numClasses));

		cls = uint32_t(bboxTargetsData[i][0]);
		if (cls > 0) {
			start = 4 * cls;
			end = start + 4;

			const vector<float>& bboxTargetData = bboxTargetsData[i];
			vector<float>& bboxTarget 			= bboxTargets[i];
			vector<float>& bboxInsideWeight 	= bboxInsideWeights[i];

			bboxTarget[start] 	  = bboxTargetData[1];
			bboxTarget[start + 1] = bboxTargetData[2];
			bboxTarget[start + 2] = bboxTargetData[3];
			bboxTarget[start + 3] = bboxTargetData[4];

			bboxInsideWeight[start] 	= TRAIN_BBOX_INSIDE_WEIGHTS[0];
			bboxInsideWeight[start + 1] = TRAIN_BBOX_INSIDE_WEIGHTS[1];
			bboxInsideWeight[start + 2] = TRAIN_BBOX_INSIDE_WEIGHTS[2];
			bboxInsideWeight[start + 3] = TRAIN_BBOX_INSIDE_WEIGHTS[3];
		}
	}
}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* ProposalTargetLayer<Dtype>::initLayer() {
	ProposalTargetLayer* layer = NULL;
	SNEW(layer, ProposalTargetLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void ProposalTargetLayer<Dtype>::destroyLayer(void* instancePtr) {
    ProposalTargetLayer<Dtype>* layer = (ProposalTargetLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void ProposalTargetLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 2);
	} else {
		SASSERT0(index < 5);
	}

    ProposalTargetLayer<Dtype>* layer = (ProposalTargetLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ProposalTargetLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ProposalTargetLayer<Dtype>* layer = (ProposalTargetLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ProposalTargetLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	ProposalTargetLayer<Dtype>* layer = (ProposalTargetLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void ProposalTargetLayer<Dtype>::backwardTensor(void* instancePtr) {
	ProposalTargetLayer<Dtype>* layer = (ProposalTargetLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void ProposalTargetLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template<typename Dtype>
bool ProposalTargetLayer<Dtype>::checkShape(vector<TensorShape> inputShape,
        vector<TensorShape> &outputShape) {

	if (inputShape.size() != 2)
        return false;

    const int numClasses = SLPROP(ProposalTarget, numClasses);

    // sampled rois (0, x1, y1, x2, y2)
    TensorShape outputShape1;
    outputShape1.N = 1;
    outputShape1.C = 1;
    outputShape1.H = 1;
    outputShape1.W = 5;
    outputShape.push_back(outputShape1);

    // labels
    TensorShape outputShape2;
    outputShape2.N = 1;
    outputShape2.C = 1;
    outputShape2.H = 1;
    outputShape2.W = 1;
    outputShape.push_back(outputShape2);

    // bbox_targets
    TensorShape outputShape3;
    outputShape3.N = 1;
    outputShape3.C = 1;
    outputShape3.H = 1;
    outputShape3.W = numClasses * 4;
    outputShape.push_back(outputShape3);

    // bbox_inside_weights
    TensorShape outputShape4;
    outputShape4.N = 1;
    outputShape4.C = 1;
    outputShape4.H = 1;
    outputShape4.W = numClasses * 4;
    outputShape.push_back(outputShape4);

    // bbox_outside_weights
    TensorShape outputShape5;
    outputShape5.N = 1;
    outputShape5.C = 1;
    outputShape5.H = 1;
    outputShape5.W = numClasses * 4;
    outputShape.push_back(outputShape5);

    return true;
}

template<typename Dtype>
uint64_t ProposalTargetLayer<Dtype>::calcGPUSize(vector<TensorShape> inputShape) {
    return 0UL;
}

template class ProposalTargetLayer<float>;
