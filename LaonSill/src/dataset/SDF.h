/*
 * SDF.h
 *
 *  Created on: Jun 28, 2017
 *      Author: jkim
 */

#ifndef SDF_H_
#define SDF_H_

#include <fstream>
#include <string>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/version.hpp>

#include "Datum.h"
#include "SysLog.h"

enum Mode { READ, NEW };


#define FORMAT_LEN_STR	32
#define FORMAT_LEN_INT	10
#define FORMAT_LEN_LONG	20



inline std::string format_str(const std::string& str, long length = 0) {
	const int strLength = str.length();
	if (strLength >= length) {
		return str;
	}

	std::string leading(length - strLength, '.');
	return leading + str;
}

inline std::string unformat_str(const std::string& str) {
	const int strLength = str.length();
	int startPos = 0;

	const char* ptr = str.c_str();
	for (int i = 0; i < strLength; i++) {
		if (ptr[i] != '.') {
			break;
		}
		startPos++;
	}

	// '=' for zero length string (empty string)
	SASSERT0(startPos <= strLength);
	return str.substr(startPos, strLength);
}




inline std::string format_int(int n, long numberOfLeadingZeros = 0) {
	std::ostringstream s;
	s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
	return s.str();
}

inline long unformat_int(const std::string& format_str) {
	const char* ptr = format_str.c_str();
	long result = 0;
	for (int i = 0; i < format_str.length(); i++) {
		result += (ptr[format_str.length() - i - 1] - '0') * std::pow(10, i);
	}
	return result;
}







template <typename T>
void printArray(std::vector<T>& array) {
	for (int i = 0; i < array.size(); i++) {
		std::cout << array[i] << ",";
	}
	std::cout << std::endl;
}














class LabelItem {
public:
	LabelItem() : label(0) {}

	void print() {
		std::cout << "LabelItem: " 		<< this->name 			<< std::endl;
		std::cout << "\tlabel: " 		<< this->label 			<< std::endl;
		std::cout << "\tdisplay_name: " << this->displayName	<< std::endl;
		if (this->color.size() > 0) {
			std::cout << "\tcolor: [" << this->color[0] << ", " << this->color[1] << ", " <<
					this->color[2] << "]" << std::endl;
		}
	}

	bool operator==(const LabelItem& other) {
		return (this->name == other.name &&
				this->label == other.label &&
				this->displayName == other.displayName &&
				this->color == other.color);
	}

public:
	std::string name;
	int label;
	std::string displayName;
	std::vector<int> color;


protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {

		// read (input)
		if (dynamic_cast<boost::archive::text_iarchive*>(&ar)) {
			std::string name;
			std::string label;
			std::string displayName;
			std::vector<std::string> color;

			ar & name;
			ar & label;
			ar & displayName;
			ar & color;

			this->name = unformat_str(name);
			this->label = unformat_int(label);
			this->displayName = unformat_str(displayName);
			this->color.clear();
			for (int i = 0; i < color.size(); i++) {
				this->color.push_back(unformat_int(color[i]));
			}
		}
		// write (output)
		else if (dynamic_cast<boost::archive::text_oarchive*>(&ar)) {
			std::string name = format_str(this->name, FORMAT_LEN_STR);
			std::string label = format_int(this->label, FORMAT_LEN_INT);
			std::string displayName = format_str(this->displayName, FORMAT_LEN_STR);
			std::vector<std::string> color;
			for (int i = 0; i < this->color.size(); i++) {
				color.push_back(format_int(this->color[i], 3));
			}

			ar & name;
			ar & label;
			ar & displayName;
			ar & color;
		} else {

		}
	}
};
BOOST_CLASS_VERSION(LabelItem, 0);




class SDFHeader {
public:
	SDFHeader() : numSets(0), uniform(0), channels(0), minHeight(0), minWidth(0),
	maxHeight(0), maxWidth(0), numClasses(0), size(0), version(0) {}

	// Version 확장하며 새로운 field 추가시 특히 int 변수는 반드시 초기화해줘야 함 !!!

	// version 0
	int numSets;
	std::vector<std::string> names;
	std::vector<int> setSizes;
	std::vector<long> setStartPos;
	std::vector<LabelItem> labelItemList;

	// version 1
	std::string type;
	int uniform;
	int channels;
	int minHeight;
	int minWidth;
	int maxHeight;
	int maxWidth;
	int numClasses;
	long size;				// no need to store
	uint32_t version;		// no need to store


	void init(int numSets) {
		this->numSets = numSets;
		this->names.resize(numSets, "");
		this->setSizes.resize(numSets, 0);
		this->setStartPos.resize(numSets, 0);
	}

	void print() {
		std::cout << "numSets: " << this->numSets << std::endl;
		std::cout << "names: " << std::endl;
		for (int i = 0; i < this->names.size(); i++) {
			std::cout << "\t" << this->names[i] << std::endl;
		}
		std::cout << "setSizes: " << std::endl;
		for (int i = 0; i < this->setSizes.size(); i++) {
			std::cout << "\t" << this->setSizes[i] << std::endl;
		}
		std::cout << "setStartPos: " << std::endl;
		for (int i = 0; i < this->setStartPos.size(); i++) {
			std::cout << "\t" << this->setStartPos[i] << std::endl;
		}
		std::cout << "labelItemList: " << std::endl;
		for (int i = 0; i < std::min<int>(10, this->labelItemList.size()); i++) {
			this->labelItemList[i].print();
		}
		if (this->labelItemList.size() > 10) {
			std::cout << "printed only first 10 label items ... " << std::endl;
		}
		std::cout << "type: " 		<< this->type 		<< std::endl;
		std::cout << "uniform: " 	<< this->uniform 	<< std::endl;
		std::cout << "channels: " 	<< this->channels 	<< std::endl;
		std::cout << "minHeight: " 	<< this->minHeight 	<< std::endl;
		std::cout << "minWidth: " 	<< this->minWidth	<< std::endl;
		std::cout << "maxHeight: " 	<< this->maxHeight 	<< std::endl;
		std::cout << "maxWidth: " 	<< this->maxWidth 	<< std::endl;
		std::cout << "numClasses: "	<< this->numClasses	<< std::endl;
		std::cout << "size: "		<< this->size		<< std::endl;
		std::cout << "version: "	<< this->version	<< std::endl;
	}


protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		this->version = version;

		// read (input)
		if (dynamic_cast<boost::archive::text_iarchive*>(&ar)) {
			std::string numSets;
			std::vector<std::string> names;
			std::vector<std::string> setSizes;
			std::vector<std::string> setStartPos;
			ar & numSets;
			ar & names;
			ar & setSizes;
			ar & setStartPos;
			ar & this->labelItemList;

			this->numSets = (int)unformat_int(numSets);
			this->names.clear();
			for (int i = 0; i < names.size(); i++) {
				this->names.push_back(unformat_str(names[i]));
			}
			this->setSizes.clear();
			for (int i = 0; i < setSizes.size(); i++) {
				this->setSizes.push_back((int)unformat_int(setSizes[i]));
			}
			this->setStartPos.clear();
			for (int i = 0; i < setStartPos.size(); i++) {
				this->setStartPos.push_back(unformat_int(setStartPos[i]));
			}

			if (version >= 1) {
				std::string type;
				std::string uniform;
				std::string channels;
				std::string minHeight;
				std::string minWidth;
				std::string maxHeight;
				std::string maxWidth;
				std::string numClasses;

				ar & type;
				ar & uniform;
				ar & channels;
				ar & minHeight;
				ar & minWidth;
				ar & maxHeight;
				ar & maxWidth;
				ar & numClasses;

				this->type = unformat_str(type);
				this->uniform = (int)unformat_int(uniform);
				this->channels = (int)unformat_int(channels);
				this->minHeight = (int)unformat_int(minHeight);
				this->minWidth = (int)unformat_int(minWidth);
				this->maxHeight = (int)unformat_int(maxHeight);
				this->maxWidth = (int)unformat_int(maxWidth);
				this->numClasses = (int)unformat_int(numClasses);
			}
		}
		// write (output)
		else if (dynamic_cast<boost::archive::text_oarchive*>(&ar)) {
			std::string numSets = format_int(this->numSets, FORMAT_LEN_INT);
			std::vector<std::string> names;
			for (int i = 0; i < this->names.size(); i++) {
				SASSERT0(this->names[i].length() <= FORMAT_LEN_STR);
				names.push_back(format_str(this->names[i], FORMAT_LEN_STR));
			}
			std::vector<std::string> setSizes;
			for (int i = 0; i < this->setSizes.size(); i++) {
				setSizes.push_back(format_int(this->setSizes[i], FORMAT_LEN_INT));
			}
			std::vector<std::string> setStartPos;
			for (int i = 0; i < this->setStartPos.size(); i++) {
				setStartPos.push_back(format_int(this->setStartPos[i], FORMAT_LEN_LONG));
			}

			ar & numSets;
			ar & names;
			ar & setSizes;
			ar & setStartPos;
			ar & this->labelItemList;

			if (version >= 1) {
				std::string type = format_str(this->type, FORMAT_LEN_STR);
				std::string uniform = format_int(this->uniform, FORMAT_LEN_INT);
				std::string channels = format_int(this->channels, FORMAT_LEN_INT);
				std::string minHeight = format_int(this->minHeight, FORMAT_LEN_INT);
				std::string minWidth = format_int(this->minWidth, FORMAT_LEN_INT);
				std::string maxHeight = format_int(this->maxHeight, FORMAT_LEN_INT);
				std::string maxWidth = format_int(this->maxWidth, FORMAT_LEN_INT);
				std::string numClasses = format_int(this->numClasses, FORMAT_LEN_INT);

				ar & type;
				ar & uniform;
				ar & channels;
				ar & minHeight;
				ar & minWidth;
				ar & maxHeight;
				ar & maxWidth;
				ar & numClasses;
			}
		} else {

		}
	}
};
BOOST_CLASS_VERSION(SDFHeader, 1);



/***************
 * SDF에 boost를 통해 serialize되는 신규 class의 객체가 있는 경우,
 * header 이전에 반드시 dummy로 한 번 serialize해주어야 한다.
 * -> SDFHeader, LabelItem이 적용되어 있는 상태
 */
class SDF {
public:
	SDF(const std::string& source, const Mode mode);
	SDF(const SDF& sdf);
	virtual ~SDF();

	void open();
	void close();

	void initHeader(SDFHeader& header);
	void updateHeader(SDFHeader& header);
	SDFHeader getHeader();
	long getCurrentPos();
	void setCurrentPos(long currentPos);
	void selectDataSet(const std::string& dataSet);
	void selectDataSet(const int dataSetIdx);
	const std::string& curDataSet();

	// SDF가 특정 이름의 dataSet을 갖고 있는지 확인
	int findDataSet(const std::string& dataSet);

	void put(const std::string& value);
	void commit();

	const std::string getNextValue();




	static SDFHeader retrieveSDFHeader(const std::string& source);


private:
	void update_dataset_idx_map();
	void sdf_open();
	void sdf_close();

public:
	std::string source;
	Mode mode;

private:
	SDFHeader header;
	int headerStartPos;
	int bodyStartPos;

	std::vector<std::string> values;

	size_t dbSize;

	std::ifstream ifs;
	std::ofstream ofs;
	boost::archive::text_iarchive* ia;
	boost::archive::text_oarchive* oa;

	std::vector<long> currentPos;
	std::map<std::string, int> dataSetIdxMap;
	int curDataSetIdx;


	static const std::string DATA_NAME;
	static const std::string LOCK_NAME;
	static const std::string SDF_STRING;

};

#endif /* SDF_H_ */
