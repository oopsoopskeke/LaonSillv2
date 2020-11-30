/*
 * PascalVOC.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef PASCALVOC_H_
#define PASCALVOC_H_


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <map>
#include <vector>
#include <string>
#include <fstream>

#include "frcnn_common.h"
#include "ssd_common.h"
#include "tinyxml2/tinyxml2.h"
#include "IMDB.h"

#define PASCALVOC_LOG 0


class PascalVOC : public IMDB {
public:
	PascalVOC(const std::string& imageSet, const std::string& name,
			const std::string& dataPath, const std::string& labelMapPath,
			const std::vector<float>& pixelMeans)
        : IMDB(name + "_" + imageSet),
          labelMap(labelMapPath) {

		//this->year = year;
		this->imageSet = imageSet;
		//this->devkitPath = devkitPath;
		//this->dataPath = devkitPath + "/VOC" + year;
		this->dataPath = dataPath;
		this->pixelMeans = pixelMeans;
		//buildClassToInd();

		this->labelMap.build();
		this->labelMap.printLabelItemList();



		this->imageExt = ".jpg";
		loadImageSetIndex();
	}

	/*
	void buildClassToInd() {

		this->classes = {
				"__background__",
				"aeroplane", "bicycle", "bird", "boat",
				"bottle", "bus", "car", "cat", "chair",
				"cow", "diningtable", "dog", "horse",
				"motorbike", "person", "pottedplant",
				"sheep", "sofa", "train", "tvmonitor"
		};


		for (uint32_t i = 0; i < numClasses; i++) {
			printf("Label [%02d]: %s\n", i, this->classes[i].c_str());
			this->classToInd[this->classes[i]] = i;
			this->indToClass[i] = this->classes[i];
		}
	}
	*/

	void loadImageSetIndex() {
		std::string imageSetFile =
            this->dataPath + "/ImageSets/Main/" + this->imageSet + ".txt";

		std::ifstream ifs(imageSetFile.c_str(), std::ios::in);
		if (!ifs.is_open()) {
			std::cout << "no such file: " << imageSetFile << std::endl;
			exit(1);
		}

		std::stringstream strStream;
		strStream << ifs.rdbuf();

		ifs.close();

		char line[256];
		uint32_t count = 0;
		while (!strStream.eof()) {
			strStream.getline(line, 256);
			if (strlen(line) < 1) {
				continue;
			}
			imageIndex.push_back(std::string(line));
		}
#if PASCALVOC_LOG
		const uint32_t numTrainval = imageIndex.size();
		std::cout << "numTrainval: " << numTrainval << std::endl;
		for (uint32_t i = 0; i < numTrainval; i++) {
			std::cout << imageIndex[i] << std::endl;
		}
#endif
	}

	void loadPascalAnnotation(const uint32_t index, RoIDB& roidb) {
		const std::string filename =
            this->dataPath + "/Annotations/" + imageIndex[index] + ".xml";
		Annotation annotation;
		readAnnotation(filename, annotation);

		//annotation.print();

		roidb.image = imagePathAt(index);
		roidb.width = annotation.size.width;
		roidb.height = annotation.size.height;

		roidb.mat = cv::imread(roidb.image);
		roidb.mat.convertTo(roidb.mat, CV_32F);

		float* imPtr = (float*)roidb.mat.data;

		int n = roidb.mat.rows * roidb.mat.cols * roidb.mat.channels();
		for (int i = 0; i < n; i+=3) {
			imPtr[i+0] -= this->pixelMeans[0];
			imPtr[i+1] -= this->pixelMeans[1];
			imPtr[i+2] -= this->pixelMeans[2];
		}

		const uint32_t numObjs = annotation.objects.size();

		roidb.boxes.resize(numObjs);
		roidb.gt_classes.resize(numObjs);
		roidb.gt_overlaps.resize(numObjs);

		for (uint32_t i = 0; i < numObjs; i++) {
			// boxes
			roidb.boxes[i].resize(4);
			roidb.boxes[i][0] = std::max(uint32_t(1), annotation.objects[i].xmin) - 1;	// xmin
			roidb.boxes[i][1] = std::max(uint32_t(1), annotation.objects[i].ymin) - 1;	// ymin
			roidb.boxes[i][2] = std::max(uint32_t(1), annotation.objects[i].xmax) - 1;	// xmax
			roidb.boxes[i][3] = std::max(uint32_t(1), annotation.objects[i].ymax) - 1;	// ymax

			// gt_classes
			roidb.gt_classes[i] = annotation.objects[i].label;

			// overlaps
			roidb.gt_overlaps[i].resize(this->numClasses);
			roidb.gt_overlaps[i][roidb.gt_classes[i]] = 1.0;
		}
		roidb.flipped = false;

		// max_classes
		roidb.max_classes = roidb.gt_classes;

		// max_overlaps
		np_maxByAxis(roidb.gt_overlaps, roidb.max_overlaps);

#if PASCALVOC_LOG
		roidb.print();
#endif

		// XXX: gt_overlaps의 경우 sparse matrix로 변환될 필요가 있음.
	}


	void readAnnotation(const std::string& filename, Annotation& annotation) {
		tinyxml2::XMLDocument annotationDocument;
		tinyxml2::XMLNode* annotationNode;

		annotationDocument.LoadFile(filename.c_str());
		annotationNode = annotationDocument.FirstChild();

		// filename
		tinyxml2::XMLElement* filenameElement = annotationNode->FirstChildElement("filename");
		annotation.filename = filenameElement->GetText();

		// size
		tinyxml2::XMLElement* sizeElement = annotationNode->FirstChildElement("size");
		sizeElement->FirstChildElement("width")->QueryIntText((int*)&annotation.size.width);
		sizeElement->FirstChildElement("height")->QueryIntText((int*)&annotation.size.height);
		sizeElement->FirstChildElement("depth")->QueryIntText((int*)&annotation.size.depth);

		// object
		for (tinyxml2::XMLElement* objectElement = 
                annotationNode->FirstChildElement("object"); objectElement != 0;
				objectElement = objectElement->NextSiblingElement("object")) {
			Object object;
			object.name = objectElement->FirstChildElement("name")->GetText();
			object.label = convertClassToInd(object.name);
			//object.label = atoi(objectElement->FirstChildElement("name")->GetText());
			//object.name = convertIndToClass(object.label);
			objectElement->FirstChildElement("difficult")
                         ->QueryIntText((int*)&object.difficult);

			tinyxml2::XMLElement* bndboxElement = objectElement->FirstChildElement("bndbox");
			bndboxElement->FirstChildElement("xmin")->QueryIntText((int*)&object.xmin);
			bndboxElement->FirstChildElement("ymin")->QueryIntText((int*)&object.ymin);
			bndboxElement->FirstChildElement("xmax")->QueryIntText((int*)&object.xmax);
			bndboxElement->FirstChildElement("ymax")->QueryIntText((int*)&object.ymax);

			if (!object.difficult) {
				annotation.objects.push_back(object);
			}
		}
#if PASCALVOC_LOG
		annotation.print();
#endif
	}


	uint32_t convertClassToInd(const std::string& cls) {
		return this->labelMap.convertLabelToInd(cls);

		/*
		std::map<std::string, uint32_t>::iterator itr = classToInd.find(cls);
		if (itr == classToInd.end()) {
			std::cout << "invalid class: " << cls << std::endl;
			exit(1);
		}
		return itr->second;
		*/
	}

	std::string convertIndToClass(const uint32_t ind) {
		return this->labelMap.convertIndToLabel(ind);
		/*
		std::map<uint32_t, std::string>::iterator itr = indToClass.find(ind);
		if (itr == indToClass.end()) {
			std::cout << "invalid class ind: " << ind << std::endl;
			exit(1);
		}
		return itr->second;
		*/
	}

	void getWidths(std::vector<uint32_t>& widths) {
		widths = this->widths;
	}

	void loadGtRoidb() {
		const uint32_t numImageIndex = this->imageIndex.size();
		for (uint32_t i = 0; i < numImageIndex; i++) {
			RoIDB roidb;
			loadPascalAnnotation(i, roidb);
			this->roidb.push_back(roidb);
		}
		// XXX: gtRoidb를 dump to file
	}

	std::string imagePathAt(const uint32_t index) {
		return imagePathFromIndex(index);
	}

private:
	std::string imagePathFromIndex(const uint32_t index) {
		return this->dataPath + "/JPEGImages/" + imageIndex[index] + this->imageExt;
	}

	//IMDB imdb;
	//std::string year;
	std::string imageSet;
	std::string dataPath;
	//std::map<std::string, uint32_t> classToInd;
	//std::map<uint32_t, std::string> indToClass;
	std::string imageExt;
	//std::vector<std::string> imageIndex;
	std::vector<uint32_t> widths;

	const uint32_t numClasses = 21;
	//std::vector<std::string> classes;

	//std::vector<cv::Mat> matList;

	std::vector<float> pixelMeans;

	LabelMap<float> labelMap;
};

#endif /* PASCALVOC_H_ */
