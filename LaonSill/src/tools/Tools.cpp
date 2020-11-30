#include <string>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/variant.hpp>

#include "jsoncpp/json/json.h"
#include "Tools.h"
#include "SysLog.h"
#include "Datum.h"
#include "IO.h"
#include "DataReader.h"
#include "ssd_common.h"
#include "ParamManipulator.h"
#include "DataReader.h"
#include "Datum.h"
#include "MemoryMgmt.h"

using namespace std;
namespace fs = ::boost::filesystem;



uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


void parse_label_map(const string& labelMapPath, vector<LabelItem>& labelItemList) {
	filebuf fb;
	if (fb.open(labelMapPath.c_str(), ios::in) == NULL) {
		SASSERT(false, "cannot open cluster configuration file. file path=%s",
			labelMapPath.c_str());
	}
	istream is(&fb);

	Json::Reader reader;
	Json::Value root;
	bool parse = reader.parse(is, root);

	if (!parse) {
		SASSERT(false, "invalid json-format file. file path=%s. error message=%s",
			labelMapPath.c_str(), reader.getFormattedErrorMessages().c_str());
	}

	const int numLabelItem = root.size();
	labelItemList.resize(numLabelItem);
	for (int i = 0; i < numLabelItem; i++) {
		Json::Value& item = root[i];
		SASSERT0(item.isMember("name"));
		SASSERT0(item.isMember("label"));
		SASSERT0(item.isMember("display_name"));

		labelItemList[i].name = item["name"].asString();
		labelItemList[i].label = item["label"].asInt();
		labelItemList[i].displayName = item["display_name"].asString();

		if (item.isMember("color")) {
			// color should be 3 value array (r, g, b)
			SASSERT0(item["color"].size() == 3);
			labelItemList[i].color.push_back(item["color"][0].asInt());
			labelItemList[i].color.push_back(item["color"][1].asInt());
			labelItemList[i].color.push_back(item["color"][2].asInt());
		}
	}

	for (int i = 0; i < labelItemList.size(); i++) {
		labelItemList[i].print();
	}
}

// json 파일이 labelmap 포맷인지 확인
int isLabelMapFileValid(const string& labelMapFilePath) {
	filebuf fb;
	if (fb.open(labelMapFilePath.c_str(), ios::in) == NULL) {
		return -1;
	}
	istream is(&fb);
	Json::Reader reader;
	Json::Value root;
	if (!reader.parse(is, root)) return -1;
	const int numLabelItem = root.size();
	for (int i = 0; i < numLabelItem; i++) {
		Json::Value& item = root[i];
		if (!item.isMember("name") ||
				!item.isMember("label") ||
				!item.isMember("display_name")) {
			return -1;
		}
		if (item.isMember("color") &&
				(!item["color"].size() == 0 && !item["color"].size() == 3)) {
			return -1;
		}
	}
	return 0;
}

int isMnistFileValid(const string& mnistFilePath, const int magicNumber) {
	 // Open files
	ifstream mnistFile(mnistFilePath, ios::in | ios::binary);
	if (!mnistFile) {
		return -1;
	}
	uint32_t magic;
	mnistFile.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	if (magic != magicNumber) {
		return -1;
	}
	return 0;
}




void BaseConvertParam::validityCheck() {
	// 1. labelMapFilePath check
	if (!this->labelMapFilePath.empty()) {
		// 1. 파일이 존재해야 함
		if (!fs::exists(this->labelMapFilePath)) {
			this->resultCode = -1;
			this->resultMsg = "labelMapFilePath not exists: " + this->labelMapFilePath;
			return;
		}
		// 2. 확장자가 json이어야 함
		if (fs::extension(this->labelMapFilePath) != ".json") {
			this->resultCode = -1;
			this->resultMsg = "labelMapFile should be json file: " + this->labelMapFilePath;
			return;
		}
		// 3. 유효한 labelmap json format이어야 함
		if (isLabelMapFileValid(this->labelMapFilePath) < 0) {
			this->resultCode = -1;
			this->resultMsg = "labelMapFile json format is invalid: " + this->labelMapFilePath;
			return;
		}
	}

	// 2. outFilepath check
	if (this->outFilePath.empty()) {
		this->resultCode = -1;
		this->resultMsg = "outFilePath should be specified ... ";
		return;
	}
}

void MnistDataSet::validityCheck(int& resultCode , string& resultMsg) {
	if (this->name.empty()) {
		resultCode = -1;
		resultMsg = "Mnist Data Set should have name ... ";
		return;
	}

	if (isMnistFileValid(this->imageFilePath, 2051)) {
		resultCode = -1;
		resultMsg = "Invalid Mnist Image File: " + this->imageFilePath;
		return;
	}

	if (isMnistFileValid(this->labelFilePath, 2049)) {
		resultCode = -1;
		resultMsg = "Invalid Mnist label File: " + this->labelFilePath;
		return;
	}
}

void ConvertMnistDataParam::validityCheck() {
	BaseConvertParam::validityCheck();
	if (this->resultCode < 0) {
		return;
	}

	// 1. DataSet이 최소 1개 이상이어야 함.
	if (this->dataSetList.size() < 1) {
		this->resultCode = -1;
		this->resultMsg = "At least 1 Data Set should be specified ... ";
		return;
	}

	// 2. 각 DataSet이 유효해야 함.
	for (int i = 0; i < this->dataSetList.size(); i++) {
		this->dataSetList[i].validityCheck(this->resultCode, this->resultMsg);
		if (this->resultCode < 0) {
			return;
		}
	}
	return;
}

void ImageSet::validityCheck(int& resultCode, string& resultMsg) {
	if (this->name.empty()) {
		resultCode = -1;
		resultMsg = "Image Set should have name ... ";
		return;
	}
	if (!fs::is_regular_file(this->dataSetPath)) {
		resultCode = -1;
		resultMsg = "Invalid Data Set path: " + this->dataSetPath;
		return;
	}
	return;
}

void ConvertImageSetParam::validityCheck() {
	BaseConvertParam::validityCheck();
	if (this->resultCode < 0) {
		return;
	}

	std::transform(this->encodeType.begin(), this->encodeType.end(),
			this->encodeType.begin(), ::tolower);

	if (this->encoded) {
		if (!(this->encodeType == "jpg") && !(this->encodeType == "png")) {
			this->resultCode = -1;
			this->resultMsg = "encodeType should be 'jpg' or 'png': " + this->encodeType;
			return;
		}
	}

	if (this->basePath.empty()) {
		this->resultCode = -1;
		this->resultMsg = "basePath is empty ... ";
		return;
	}

	if (this->imageSetList.size() < 1) {
		this->resultCode = -1;
		this->resultMsg = "At least 1 Image Set should be specified ... ";
		return;
	}

	for (int i = 0; i < this->imageSetList.size(); i++) {
		this->imageSetList[i].validityCheck(this->resultCode, this->resultMsg);
		if (this->resultCode < 0) {
			return;
		}
	}

	if (this->labelMapFilePath.empty() && this->numClasses == 0) {
		this->resultCode = -1;
		this->resultMsg = "One of labelMapFilePath and numClasses should be specified ... ";
		return;
	}

	return;
}

void ConvertAnnoSetParam::validityCheck() {
	ConvertImageSetParam::validityCheck();
	if (this->resultCode < 0) {
		return;
	}

	if (this->labelMapFilePath.empty()) {
		this->resultCode = -1;;
		this->resultMsg = "labelMapFilePath should be specified.";
		return;
	}
	return;
}















void printDenormalizeParamUsage(char* prog) {
    fprintf(stderr, "Usage: %s -i old_param_path -o new_param_path\n", prog);
    fprintf(stderr, "\t-i: old param file path\n");
	fprintf(stderr, "\t-o: new param file path\n");
    exit(EXIT_FAILURE);
}

void denormalizeTest(int argc, char** argv) {
	string old_param_path;
	string new_param_path;

	int opt;
	while ((opt = getopt(argc, argv, "i:o:")) != -1) {
		if (!optarg) {
			printDenormalizeParamUsage(argv[0]);
		}

		switch (opt) {
		case 'i':
			old_param_path = string(optarg);
			break;
		case 'o':
			new_param_path = string(optarg);
			break;
		default:
			printDenormalizeParamUsage(argv[0]);
			break;
		}
	}

	if (!old_param_path.length() || !new_param_path.length()) {
		printDenormalizeParamUsage(argv[0]);
	}

	cout << endl;
	cout << "::: Denormalize Param Configurations :::" << endl;
	cout << "old_param_path: " << old_param_path << endl;
	cout << "new_param_path: " << new_param_path << endl;
	cout << ":::::::::::::::::::::::::::::::::::::::::" << endl;
	cout << endl;

	denormalize(old_param_path, new_param_path);
}

void denormalize(const string& old_param_path, const string& new_param_path) {
	/*
	ParamManipulator<float> pm(
			"/home/jkim/Dev/SOOOA_HOME/network/frcnn_630000.param",
			"/home/jkim/Dev/SOOOA_HOME/network/frcnn_630000_dn.param");
			*/
	ParamManipulator<float> pm(old_param_path, new_param_path);
	//pm.printParamList();

	pm.denormalizeParams({"bbox_pred_weight", "bbox_pred_bias"},
			{0.f, 0.f, 0.f, 0.f},
			{0.1f, 0.1f, 0.2f, 0.2f});

	pm.save();
}



std::string format_int(int n, int numberOfLeadingZeros = 0) {
	//std::ostringstream s;
	//s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
	//return s.str();
	return "00000000";
}



void printConvertMnistDataUsage(char* prog) {
    fprintf(stderr, "Usage: %s -i image_file -l label_file -o db_path\n", prog);
    fprintf(stderr, "\t-i: image file path\n");
	fprintf(stderr, "\t-l: label file path\n");
	fprintf(stderr, "\t-o: output db file path\n");
    exit(EXIT_FAILURE);
}


void convertMnistDataTest(int argc, char** argv) {
	string image_filename;
	string label_filename;
	string db_path;

	int opt;
	while ((opt = getopt(argc, argv, "i:l:o:")) != -1) {
		if (!optarg) {
			printConvertMnistDataUsage(argv[0]);
		}

		switch (opt) {
		case 'i':
			image_filename = string(optarg);
			break;
		case 'l':
			label_filename = string(optarg);
			break;
		case 'o':
			db_path = string(optarg);
			break;
		default:
			printConvertMnistDataUsage(argv[0]);
			break;
		}
	}

	if (image_filename.empty() ||
			label_filename.empty() ||
			db_path.empty()) {
		printConvertMnistDataUsage(argv[0]);
	}

	cout << endl;
	cout << "::: Convert Mnist Data Configurations :::" << endl;
	cout << "image_filename: " << image_filename << endl;
	cout << "label_filename: " << label_filename << endl;
	cout << "db_path: " << db_path << endl;
	cout << ":::::::::::::::::::::::::::::::::::::::::" << endl;
	cout << endl;

    //const string image_filename = "/home/jkim/Dev/git/caffe/data/mnist/train-images-idx3-ubyte";
    //const string label_filename = "/home/jkim/Dev/git/caffe/data/mnist/train-labels-idx1-ubyte";
    //const string db_path = "/home/jkim/imageset/lmdb/mnist_train_lmdb/";

	//convertMnistData(image_filename, label_filename, db_path);

}


// VERY TEMPORAL
void convertMnistDataTemp(ConvertMnistDataParam& param) {
	param.validityCheck();
	if (param.resultCode < 0) {
		return;
	}

	int numDataSets = param.dataSetList.size();

	SDFHeader header;
	header.init(numDataSets);
	header.type = "DATUM";
	header.uniform = 1;
	header.numClasses = 8;

	if (!param.labelMapFilePath.empty()) {
		parse_label_map(param.labelMapFilePath, header.labelItemList);
	}


	SDF sdf(param.outFilePath, Mode::NEW);
	sdf.open();
	sdf.initHeader(header);
	int imgCount[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	for (int dataSetIdx = 0; dataSetIdx < param.dataSetList.size(); dataSetIdx++) {
		const MnistDataSet& dataSet = param.dataSetList[dataSetIdx];
		header.names[dataSetIdx] = dataSet.name;
		header.setStartPos[dataSetIdx] = sdf.getCurrentPos();
		const string& imageFilePath = dataSet.imageFilePath;
		const string& labelFilePath = dataSet.labelFilePath;

		 // Open files
		ifstream image_file(imageFilePath, ios::in | ios::binary);
		ifstream label_file(labelFilePath, ios::in | ios::binary);
		if (!image_file) {
			cout << "Unable to open file " << imageFilePath << endl;
			SASSERT0(false);
		}
		if (!label_file) {
			cout << "Unable to open file " << labelFilePath << endl;
			SASSERT0(false);
		}
		// Read the magic and the meta data
		uint32_t magic;
		uint32_t num_items;
		uint32_t num_labels;
		uint32_t rows;
		uint32_t cols;

		image_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2051) {
			cout << "Incorrect image file magic." << endl;
			SASSERT0(false);
		}
		label_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2049) {
			cout << "Incorrect label file magic." << endl;
			SASSERT0(false);
		}
		image_file.read(reinterpret_cast<char*>(&num_items), 4);
		num_items = swap_endian(num_items);
		label_file.read(reinterpret_cast<char*>(&num_labels), 4);
		num_labels = swap_endian(num_labels);
		SASSERT0(num_items == num_labels);
		image_file.read(reinterpret_cast<char*>(&rows), 4);
		rows = swap_endian(rows);
		image_file.read(reinterpret_cast<char*>(&cols), 4);
		cols = swap_endian(cols);



		// Storing to db
		char label;
		char* pixels = NULL;
		SMALLOC(pixels, char, rows * cols * sizeof(char));
		SASSUME0(pixels != NULL);
		int count = 0;
		string value;

		Datum datum;
		datum.channels = 1;
		datum.height = rows;
		datum.width = cols;
		header.channels = 1;
		header.minHeight = rows;
		header.minWidth = cols;
		header.maxHeight = rows;
		header.maxWidth = cols;

		cout << "A total of " << num_items << " items." << endl;
		cout << "Rows: " << rows << " Cols: " << cols << endl;

		//sdf.put("num_data", std::to_string(num_items));
		//sdf.commit();
		//header.setSizes[dataSetIdx] = num_items;

		//string buffer(rows * cols, ' ');
		for (int item_id = 0; item_id < num_items; ++item_id) {
			image_file.read(pixels, rows * cols);
			label_file.read(&label, 1);

			imgCount[label]++;
			if (label >= header.numClasses) {
				continue;
			}

			//for (int i = 0; i < rows*cols; i++) {
			//   buffer[i] = pixels[i];
			//}
			//datum.data = buffer;
			datum.data.assign(reinterpret_cast<const char*>(pixels), rows * cols);
			datum.label = label;
			value = serializeToString(&datum);

			sdf.put(value);

			if (++count % 1000 == 0) {
				sdf.commit();
				cout << "Processed " << count << " files." << endl;
			}
		}
		// write the last batch

		if (count % 1000 != 0) {
			sdf.commit();
			cout << "Processed " << count << " files." << endl;
		}

		header.setSizes[dataSetIdx] = count;
		SDELETE(pixels);
	}

	sdf.updateHeader(header);
	sdf.close();

	cout << "number of images for each number" << endl;
	int total = 0;
	for (int i = 0; i < 10; i++) {
		total += imgCount[i];
		cout << i << ": " << imgCount[i] << ", " << total << endl;

	}

	for (int i = 0; i < header.numSets; i++) {
		if (header.setSizes[i] == 0) {
			param.resultCode = -1;
			param.resultMsg = "one of data set size is 0 ... ";
		}
	}
}



void convertMnistData(ConvertMnistDataParam& param) {
	param.validityCheck();
	if (param.resultCode < 0) {
		return;
	}

	int numDataSets = param.dataSetList.size();

	SDFHeader header;
	header.init(numDataSets);
	header.type = "DATUM";
	header.uniform = 1;
	header.numClasses = 10;

	if (!param.labelMapFilePath.empty()) {
		parse_label_map(param.labelMapFilePath, header.labelItemList);
	}


	SDF sdf(param.outFilePath, Mode::NEW);
	sdf.open();
	sdf.initHeader(header);

	for (int dataSetIdx = 0; dataSetIdx < param.dataSetList.size(); dataSetIdx++) {
		const MnistDataSet& dataSet = param.dataSetList[dataSetIdx];
		header.names[dataSetIdx] = dataSet.name;
		header.setStartPos[dataSetIdx] = sdf.getCurrentPos();
		const string& imageFilePath = dataSet.imageFilePath;
		const string& labelFilePath = dataSet.labelFilePath;

		 // Open files
		ifstream image_file(imageFilePath, ios::in | ios::binary);
		ifstream label_file(labelFilePath, ios::in | ios::binary);
		if (!image_file) {
			cout << "Unable to open file " << imageFilePath << endl;
			SASSERT0(false);
		}
		if (!label_file) {
			cout << "Unable to open file " << labelFilePath << endl;
			SASSERT0(false);
		}
		// Read the magic and the meta data
		uint32_t magic;
		uint32_t num_items;
		uint32_t num_labels;
		uint32_t rows;
		uint32_t cols;

		image_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2051) {
			cout << "Incorrect image file magic." << endl;
			SASSERT0(false);
		}
		label_file.read(reinterpret_cast<char*>(&magic), 4);
		magic = swap_endian(magic);
		if (magic != 2049) {
			cout << "Incorrect label file magic." << endl;
			SASSERT0(false);
		}
		image_file.read(reinterpret_cast<char*>(&num_items), 4);
		num_items = swap_endian(num_items);
		label_file.read(reinterpret_cast<char*>(&num_labels), 4);
		num_labels = swap_endian(num_labels);
		SASSERT0(num_items == num_labels);
		image_file.read(reinterpret_cast<char*>(&rows), 4);
		rows = swap_endian(rows);
		image_file.read(reinterpret_cast<char*>(&cols), 4);
		cols = swap_endian(cols);



		// Storing to db
		char label;
		char* pixels = NULL;
		SMALLOC(pixels, char, rows * cols * sizeof(char));
		SASSUME0(pixels != NULL);
		int count = 0;
		string value;

		Datum datum;
		datum.channels = 1;
		datum.height = rows;
		datum.width = cols;
		header.channels = 1;
		header.minHeight = rows;
		header.minWidth = cols;
		header.maxHeight = rows;
		header.maxWidth = cols;

		cout << "A total of " << num_items << " items." << endl;
		cout << "Rows: " << rows << " Cols: " << cols << endl;

		//sdf.put("num_data", std::to_string(num_items));
		//sdf.commit();
		header.setSizes[dataSetIdx] = num_items;

		//string buffer(rows * cols, ' ');
		for (int item_id = 0; item_id < num_items; ++item_id) {
			image_file.read(pixels, rows * cols);
			label_file.read(&label, 1);

			//for (int i = 0; i < rows*cols; i++) {
			//   buffer[i] = pixels[i];
			//}
			//datum.data = buffer;
			datum.data.assign(reinterpret_cast<const char*>(pixels), rows * cols);
			datum.label = label;
			value = serializeToString(&datum);

			sdf.put(value);

			if (++count % 1000 == 0) {
				sdf.commit();
				cout << "Processed " << count << " files." << endl;
			}
		}
		// write the last batch

		if (count % 1000 != 0) {
			sdf.commit();
			cout << "Processed " << count << " files." << endl;
		}
		SDELETE(pixels);
	}

	sdf.updateHeader(header);
	sdf.close();

	for (int i = 0; i < header.numSets; i++) {
		if (header.setSizes[i] == 0) {
			param.resultCode = -1;
			param.resultMsg = "one of data set size is 0 ... ";
		}
	}
}







void printConvertImageSetUsage(char* prog) {
    fprintf(stderr, "Usage: %s [-g | -s | -m | -c | -w resize_width | -h resize_height | -l dataset_path] -i image_path -o output_path\n", prog);
    fprintf(stderr, "\t-g: gray image input\n");
    fprintf(stderr, "\t-s: shuffle the image set\n");
    fprintf(stderr, "\t-m: multiple labels\n");
    fprintf(stderr, "\t-c: channel not separated\n");
    fprintf(stderr, "\t-w: resize image with specified width\n");
    fprintf(stderr, "\t-h: resize image with specified height\n");
    fprintf(stderr, "\t-i: image path\n");
    fprintf(stderr, "\t-d: dataset path\n");
    fprintf(stderr, "\t-o: output path\n");

    exit(EXIT_FAILURE);
}


void convertImageSetTest(int argc, char** argv) {
	string argv1;
	string argv2;
	string argv3;

	bool 	FLAGS_gray = false;
	bool 	FLAGS_shuffle = false;		// default false
	bool	FLAGS_multi_label = false;
	bool	FLAGS_channel_separated = true;
	int 	FLAGS_resize_width = 0;		// default 0
	int		FLAGS_resize_height = 0;	// default 0
	bool	FLAGS_check_size = false;
	bool	FLAGS_encoded = false;
	string	FLAGS_encode_type = "";

	//g: gray
	//s: shuffle
	//w: resize_width
	//h: resize_height
	//i: image_path
	//d: dataset
	//o: output
	int opt;
	while ((opt = getopt(argc, argv, "gsmcw:h:i:d:o:")) != -1) {
		switch (opt) {
		case 'g':
			FLAGS_gray = true;
			break;
		case 's':
			FLAGS_shuffle = true;
			break;
		case 'm':
			FLAGS_multi_label = true;
			break;
		case 'c':
			FLAGS_channel_separated = false;
			break;
		case 'w':
			if (optarg) FLAGS_resize_width = atoi(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'h':
			if (optarg) FLAGS_resize_height = atoi(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'i':
			if (optarg) argv1 = optarg;
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'd':
			if (optarg) argv2 = optarg;
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'o':
			if (optarg) argv3 = optarg;
			else printConvertImageSetUsage(argv[0]);
			break;
		default:
			printConvertImageSetUsage(argv[0]);
			break;
		}
	}

	if (argv1.empty() ||
			// assume image only mode, if dataset is not provided
			// !argv2.length() ||
			argv3.empty()) {
		printConvertImageSetUsage(argv[0]);
	}

	cout << endl;
	cout << "::: Convert Image Set Configurations :::" << endl;
	cout << "gray: " << FLAGS_gray << endl;
	cout << "shuffle: " << FLAGS_shuffle << endl;
	cout << "multi_pabel: " << FLAGS_multi_label << endl;
	cout << "channel_separated: " << FLAGS_channel_separated << endl;
	cout << "resize_width: " << FLAGS_resize_width << endl;
	cout << "resize_height: " << FLAGS_resize_height << endl;
	cout << "image_path: " << argv1 << endl;
	cout << "dataset_path: " << argv2 << endl;
	cout << "output_path: " << argv3 << endl;
	cout << "::::::::::::::::::::::::::::::::::::::::" << endl;
	cout << endl;

	//convertImageSet(FLAGS_gray, FLAGS_shuffle, FLAGS_multi_label, FLAGS_channel_separated,
	//		FLAGS_resize_width, FLAGS_resize_height, FLAGS_check_size, FLAGS_encoded,
	//		FLAGS_encode_type, argv1, argv2, argv3);

}


void updateHeaderInfo(ConvertImageSetParam* param, const string& type, bool is_color,
		int minHeight, int minWidth, int maxHeight, int maxWidth, SDFHeader& header) {

	header.type = type;
	header.channels = is_color ? 3 : 1;

	header.minHeight = minHeight;
	header.minWidth = minWidth;
	header.maxHeight = maxHeight;
	header.maxWidth = maxWidth;
	if (minHeight == maxHeight && minWidth == maxWidth) {
		header.uniform = 1;
	} else {
		header.uniform = 0;
	}

	if (header.labelItemList.size() == 0 && param->numClasses == 0) {
		param->resultCode = -1;
		param->resultMsg = "One of labelMapFilePath and numClasses should be specified ... ";
		return;
	}

	if (header.labelItemList.size() > 0 && param->numClasses > 0 &&
			header.labelItemList.size() != param->numClasses) {
		param->resultCode = -1;
		param->resultMsg = "labelMapFilePath info and numClasses are inconsistent ... ";
		return;
	}

	header.numClasses = param->numClasses > 0 ? param->numClasses : header.labelItemList.size();
}




void convertImageSet(ConvertImageSetParam& param) {
	param.validityCheck();
	if (param.resultCode < 0) {
		return;
	}

	int numImageSets = param.imageSetList.size();

	SDFHeader header;
	header.init(numImageSets);
	header.print();

	if (!param.labelMapFilePath.empty()) {
		parse_label_map(param.labelMapFilePath, header.labelItemList);
	}

	bool FLAGS_gray = param.gray;
	bool FLAGS_shuffle = param.shuffle;
	bool FLAGS_multi_label = param.multiLabel;
	bool FLAGS_channel_separated = param.channelSeparated;
	int FLAGS_resize_width = param.resizeWidth;
	int FLAGS_resize_height = param.resizeHeight;
	bool FLAGS_check_size = param.checkSize;
	bool FLAGS_encoded = param.encoded;
	const string& FLAGS_encode_type = param.encodeType;
	const string& argv1 = param.basePath;

	const string& argv3 = param.outFilePath;

	const bool is_color = !FLAGS_gray;
	const bool multi_label = FLAGS_multi_label;
	const bool channel_separated = FLAGS_channel_separated;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = !FLAGS_encode_type.empty() ? FLAGS_encode_type : "";

	// Create new DB
	SDF sdf(argv3, Mode::NEW);
	sdf.open();
	sdf.initHeader(header);

	int minHeight = INT_MAX;
	int minWidth = INT_MAX;
	int maxHeight = 0;
	int maxWidth = 0;

	for (int imageSetIdx = 0; imageSetIdx < numImageSets; imageSetIdx++) {
		const ImageSet& imageSet = param.imageSetList[imageSetIdx];
		header.names[imageSetIdx] = imageSet.name;
		header.setStartPos[imageSetIdx] = sdf.getCurrentPos();
		const string& argv2 = imageSet.dataSetPath;
		const bool image_only_mode = argv2.empty() ? true : false;

		vector<pair<string, vector<int>>> lines;
		string line;
		size_t pos;
		int label;
		vector<int> labelList;


		// image and label pairs are provided
		if (!image_only_mode) {
			ifstream infile(argv2);
			while (std::getline(infile, line)) {
				labelList.clear();
				pos = line.find_last_of(' ');

				// sinlge label is provided
				if (!multi_label) {
					labelList.push_back(atoi(line.substr(pos + 1).c_str()));
					lines.push_back(std::make_pair(line.substr(0, pos), labelList));
				}
				// multiple labels are provided
				else {
					string first = line.substr(0, pos);
					string labels = line.substr(pos + 1);
					pos = labels.find_first_of(',');
					while (pos != string::npos) {
						labelList.push_back(atoi(labels.substr(0, pos).c_str()));
						labels = labels.substr(pos + 1);
						pos = labels.find_first_of(',');
					}
					labelList.push_back(atoi(labels.substr(0, pos).c_str()));
					lines.push_back(std::make_pair(first.substr(0, first.length()), labelList));
				}
			}
		}
		// only images provided
		else {
			SASSERT(fs::exists(argv1), "image path %s not exists ... ", argv1.c_str());
			SASSERT(fs::is_directory(argv1), "image path %s is not directory ... ", argv1.c_str());

			const string ext = ".jpg";
			fs::path image_path(argv1);
			fs::recursive_directory_iterator it(image_path);
			fs::recursive_directory_iterator endit;

			int count = 0;
			while (it != endit) {
				if (fs::is_regular_file(*it) && it->path().extension() == ext) {
					string path = it->path().filename().string();
					// path를 그대로 전달할 경우 error ...
					// substr() 호출한 결과를 전달할 경우 문제 x
					labelList.push_back(0);
					lines.push_back(std::make_pair(path.substr(0, path.length()), labelList));
					//lines.push_back(std::make_pair<string, vector<int>>(path, {0}));
				}
				it++;
			}
		}

		if (FLAGS_shuffle) {
			// randomly shuffle data
			std::random_shuffle(lines.begin(), lines.end());
		} else {

		}

		cout << "A total of " << lines.size() << " images." << endl;
		/*
		for (int i = 0; i < std::min<int>(lines.size(), 100); i++) {
			cout << "fn: " << lines[i].first << ", label: " << lines[i].second << endl;
		}
		*/


		if (encode_type.size() && !encoded) {
			cout << "encode_type specified, assuming encoded=true.";
		}

		int resize_height = std::max<int>(0, FLAGS_resize_height);
		int resize_width = std::max<int>(0, FLAGS_resize_width);


		// Storing to db
		//string root_folder(argv1);
		fs::path root_folder(argv1);
		Datum datum;
		int count = 0;
		int data_size = 0;
		bool data_size_initialized = false;


		// 이 시점에서 data 수를 저장할 경우
		// 아래 status가 false인 경우 등의 상황에서 수가 정확하지 않을 가능성이 있음.
		//sdf.put("num_data", std::to_string(lines.size()));
		//sdf.commit();
		header.setSizes[imageSetIdx] = lines.size();

		for (int line_id = 0; line_id < lines.size(); line_id++) {
			bool status;
			string enc = encode_type;
			if (encoded && !enc.size()) {
				// Guess the encoding type from the file name
				string fn = lines[line_id].first;
				size_t p = fn.rfind('.');
				if (p == fn.npos) {
					cout << "Failed to guess the encoding of '" << fn << "'";
				}
				enc = fn.substr(p);
				std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
			}
			status = ReadImageToDatum((root_folder / lines[line_id].first).string(), lines[line_id].second,
					resize_height, resize_width, 0, 0, channel_separated, is_color, enc, &datum);

			if (status == false) {
				cout << lines[line_id].first << " file is corrupted ... skipping ... " << endl;
				header.setSizes[imageSetIdx]--;
				continue;
			}
			SASSERT0(!check_size);

			if (minHeight > datum.height) minHeight = datum.height;
			if (minWidth > datum.width) minWidth = datum.width;
			if (maxHeight < datum.height) maxHeight = datum.height;
			if (maxWidth < datum.width) maxWidth = datum.width;


			// Put in db
			//string out = Datum::serializeToString(&datum);
			string out = serializeToString(&datum);
			sdf.put(out);

			if (++count % 1000 == 0) {
				sdf.commit();
				cout << "Processed " << count << " files." << endl;
			}
		}

		// write the last batch
		if (count % 1000 != 0) {
			sdf.commit();
			cout << "Processed " << count << " files." << endl;
		}
	}


	updateHeaderInfo(&param, "DATUM", is_color, minHeight, minWidth,
			maxHeight, maxWidth, header);
	header.print();

	sdf.updateHeader(header);
	sdf.close();


	for (int i = 0; i < header.numSets; i++) {
		if (header.setSizes[i] == 0) {
			param.resultCode = -1;
			param.resultMsg = "one of data set size is 0 ... ";
		}
	}



}














void convertAnnoSetTest(int argc, char** argv) {
	bool 	FLAGS_gray = false;
	bool 	FLAGS_shuffle = false;		// default false
	bool	FLAGS_multi_label = false;
	bool	FLAGS_channel_separated = true;
	int 	FLAGS_resize_width = 0;		// default 0
	int		FLAGS_resize_height = 0;	// default 0
	bool	FLAGS_check_size = false;	// check that all the datum have the same size
	bool	FLAGS_encoded = true;		// default true, the encoded image will be save in datum
	string	FLAGS_encode_type = "jpg";	// default "", what type should we encode the image as ('png', 'jpg', ... )
	string	FLAGS_anno_type = "detection";		// default "classification"
	string	FLAGS_label_type = "xml";
	string	FLAGS_label_map_file = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt";	// default ""
	bool	FLAGS_check_label = true;			// default false, check that there is no duplicated name/label
	int		FLAGS_min_dim = 0;
	int		FLAGS_max_dim = 0;

	const string argv1 = "/home/jkim/Dev/git/caffe_ssd/data/VOCdevkit/"; // base path
	const string argv2 = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/trainval.txt"; // dataset ... (trainval.txt, ...)
	const string argv3 = "/home/jkim/Dev/SOOOA_HOME/data/sdf/voc2007_train_sdf/";		// sdf path
	//const string argv2 = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/test.txt"; // dataset ... (trainval.txt, ...)
	//const string argv3 = "/home/jkim/Dev/SOOOA_HOME/data/sdf/voc2007_test_sdf/";		// sdf path

	//convertAnnoSet(FLAGS_gray, FLAGS_shuffle, FLAGS_multi_label, FLAGS_channel_separated,
	//		FLAGS_resize_width, FLAGS_resize_height, FLAGS_check_size, FLAGS_encoded,
	//		FLAGS_encode_type, FLAGS_anno_type, FLAGS_label_type,
	//		FLAGS_label_map_file,
	//		FLAGS_check_label, FLAGS_min_dim, FLAGS_max_dim, argv1, argv2, argv3);

}


void convertAnnoSet(ConvertAnnoSetParam& param) {
	param.validityCheck();
	if (param.resultCode < 0) {
		return;
	}

	int numImageSets = param.imageSetList.size();

	SDFHeader header;
	header.init(numImageSets);

	if (!param.labelMapFilePath.empty()) {
		parse_label_map(param.labelMapFilePath, header.labelItemList);
	}

	bool FLAGS_gray = param.gray;
	bool FLAGS_shuffle = param.shuffle;
	bool FLAGS_multi_label = param.multiLabel;
	bool FLAGS_channel_separated = param.channelSeparated;
	int FLAGS_resize_width = param.resizeWidth;
	int FLAGS_resize_height = param.resizeHeight;
	bool FLAGS_check_size = param.checkSize;
	bool FLAGS_encoded = param.encoded;
	const string& FLAGS_encode_type = param.encodeType;
	const string& FLAGS_anno_type = param.annoType;
	const string& FLAGS_label_type = param.labelType;
	const string& FLAGS_label_map_file = param.labelMapFilePath;
	bool FLAGS_check_label = param.checkLabel;
	int FLAGS_min_dim = param.minDim;
	int FLAGS_max_dim = param.maxDim;
	const string& argv1 = param.basePath;
	const string& argv3 = param.outFilePath;

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;
	const string anno_type = FLAGS_anno_type;
	AnnotationType type;
	const string label_type = FLAGS_label_type;
	const bool check_label = FLAGS_check_label;
	//map<string, int> name_to_label;


	// Create new DB
	SDF sdf(argv3, Mode::NEW);
	sdf.open();
	sdf.initHeader(header);

	int minHeight = INT_MAX;
	int minWidth = INT_MAX;
	int maxHeight = 0;
	int maxWidth = 0;

	LabelMap<float> label_map;
	label_map.build(header.labelItemList);

	for (int imageSetIdx = 0; imageSetIdx < numImageSets; imageSetIdx++) {
		const ImageSet& imageSet = param.imageSetList[imageSetIdx];
		header.names[imageSetIdx] = imageSet.name;
		header.setStartPos[imageSetIdx] = sdf.getCurrentPos();
		const string& argv2 = imageSet.dataSetPath;

		ifstream infile(argv2);
		vector<pair<string, boost::variant<int, string>>> lines;
		string filename;
		int label;
		string labelname;
		SASSERT(anno_type == "detection", "only anno_type 'detection' is supported.");
		type = AnnotationType::BBOX;


		while (infile >> filename >> labelname) {
			lines.push_back(make_pair(filename, labelname));
		}

		if (FLAGS_shuffle) {
			// randomly shuffle data
			cout << "Shuffling data" << endl;
			//shuffle(lines.begin(), lines.end());
			std::random_shuffle(lines.begin(), lines.end());
		}
		cout << "A total of " << lines.size() << " images." << endl;

		if (encode_type.size() && !encoded) {
			cout << "encode_type specified, assuming encoded=true." << endl;
		}

		int min_dim = std::max<int>(0, FLAGS_min_dim);
		int max_dim = std::max<int>(0, FLAGS_max_dim);
		int resize_height = std::max<int>(0, FLAGS_resize_height);
		int resize_width = std::max<int>(0, FLAGS_resize_width);








		// Storing to db
		//string root_folder(argv1);
		fs::path root_folder(argv1);
		AnnotatedDatum anno_datum;
		int count = 0;
		int data_size = 0;
		bool data_size_initialized = false;

		// 이 시점에서 data 수를 저장할 경우
		// 아래 status가 false인 경우 등의 상황에서 수가 정확하지 않을 가능성이 있음.
		//sdf.put("num_data", std::to_string(lines.size()));
		//sdf.commit();
		header.setSizes[imageSetIdx] = lines.size();



		for (int line_id = 0; line_id < lines.size(); line_id++) {
			bool status = true;
			string enc = encode_type;
			if (encoded && !enc.size()) {
				// Guess the encoding type from the file name
				string fn = lines[line_id].first;
				size_t p = fn.rfind('.');
				if (p == fn.npos) {
					cout << "Failed to guess the encoding of '" << fn << "'";
				}
				enc = fn.substr(p);
				std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
			}
			filename = (root_folder / lines[line_id].first).string();
			labelname = (root_folder / boost::get<string>(lines[line_id].second)).string();
			status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
					resize_width, min_dim, max_dim, is_color, enc, type, label_type,
					label_map.labelToIndMap, &anno_datum);
			anno_datum.type = AnnotationType::BBOX;
			//anno_datum.print();

			if (status == false) {
				cout << "Failed to read " << lines[line_id].first << endl;
				header.setSizes[imageSetIdx]--;
				continue;
			}
			SASSERT0(!check_size);

			if (minHeight > anno_datum.height) minHeight = anno_datum.height;
			if (minWidth > anno_datum.width) minWidth = anno_datum.width;
			if (maxHeight < anno_datum.height) maxHeight = anno_datum.height;
			if (maxWidth < anno_datum.width) maxWidth = anno_datum.width;


			// Put in db
			//string out = Datum::serializeToString(&datum);
			string out = serializeToString(&anno_datum);
			sdf.put(out);

			if (++count % 1000 == 0) {
				sdf.commit();
				cout << "Processed " << count << " files." << endl;
			}
		}

		// write the last batch
		if (count % 1000 != 0) {
			sdf.commit();
			cout << "Processed " << count << " files." << endl;
		}
	}


	updateHeaderInfo(&param, "ANNO_DATUM", is_color, minHeight, minWidth,
			maxHeight, maxWidth, header);

	sdf.updateHeader(header);
	sdf.close();


	for (int i = 0; i < header.numSets; i++) {
		if (header.setSizes[i] == 0) {
			param.resultCode = -1;
			param.resultMsg = "one of data set size is 0 ... ";
		}
	}
}




void computeImageMeanTest(int argc, char** argv) {

	computeImageMean("/home/jkim/Dev/SOOOA_HOME/data/sdf/plantynet_train_0.25/");

}



void computeImageMean(const std::string& sdf_path) {
	DataReader<Datum> dr(sdf_path);
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	size_t mean[3] = {0, 0, 0};
	size_t elemCnt[3] = {0, 0, 0};

	int i = 0;
	while (i < numData) {
		Datum* datum = dr.getNextData();

		const int channels = datum->channels;
		const int height = datum->height;
		const int width = datum->width;
		const uchar* dataPtr = (uchar*)datum->data.c_str();

		for (int c = 0; c < channels; c++) {
			int imgArea = height * width;
			for (int idx = 0; idx < imgArea; idx++) {
				mean[c] += dataPtr[c * imgArea + idx];
			}
			elemCnt[c] += imgArea;
		}

		if (++i % 1000 == 0) {
			cout << "Processed " << i << " images." << endl;
		}
	}

	if (i % 1000 != 0) {
		cout << "Processed " << i << " images." << endl;
	}

	for (i = 0; i < 3; i++) {
		double m = 0.0;
		if (elemCnt[i] > 0) {
			m = mean[i] / (double)elemCnt[i];
		}
		cout << "sum: " << mean[i];
		cout << "\tcount: " << elemCnt[i];
		cout << "\t" << i << "th mean: " << m << endl;
	}
}




