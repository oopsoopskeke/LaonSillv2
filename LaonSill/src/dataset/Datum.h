/*
 * Datum.h
 *
 *  Created on: Jun 29, 2017
 *      Author: jkim
 */

#ifndef DATUM_H_
#define DATUM_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

#include "EnumDef.h"











class Datum {
public:
	// version 0
	int channels;
	int height;
	int width;
	int label;
	bool encoded;
	std::vector<float> float_data;
	std::string data;

	Datum()
	: channels(0), height(0), width(0), label(0), encoded(false), data("") {}

	size_t getImgSize() const {
		return this->channels * this->height * this->width;
	}

	bool hasLabel() {
		return (this->label > 0);
	}

	void print() {
		std::cout << "channels: " << this->channels << std::endl;
		std::cout << "height: " << this->height << std::endl;
		std::cout << "width: " << this->width << std::endl;
		std::cout << "label: " << this->label << std::endl;
		std::cout << "encoded: " << this->encoded << std::endl;
		std::cout << "float_data: ";
		for (int i = 0; i < this->float_data.size(); i++) {
			std::cout << this->float_data[i] << ",";
		}
		std::cout << std::endl;
	}



protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & channels;
		ar & height;
		ar & width;
		ar & label;
		ar & encoded;
		ar & float_data;
		ar & data;
	}
};
BOOST_CLASS_VERSION(Datum, 0);





// The normalized bounding box [0, 1] w.r.t. the input image size
class NormalizedBBox {
public:
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int label;
	bool difficult;
	float score;
	float size;

	NormalizedBBox() {
		this->xmin = 0.f;
		this->ymin = 0.f;
		this->xmax = 0.f;
		this->ymax = 0.f;
		this->label = 0;
		this->difficult = false;
		this->score = 0.f;
		this->size = 0.f;
	}

	bool operator==(const NormalizedBBox& other) {
		return (this->xmin == other.xmin && this->ymin == other.ymin &&
				this->xmax == other.xmax && this->ymax == other.ymax &&
				this->label == other.label && this->difficult == other.difficult &&
				this->score == other.score && this->size == other.size);
	}

	void print() {
		std::cout << "\txmin: " 		<< this->xmin		<< std::endl;
		std::cout << "\tymin: " 		<< this->ymin		<< std::endl;
		std::cout << "\txmax: " 		<< this->xmax		<< std::endl;
		std::cout << "\tymax: " 		<< this->ymax		<< std::endl;
		std::cout << "\tlabel: " 		<< this->label		<< std::endl;
		std::cout << "\tdifficult: "	<< this->difficult	<< std::endl;
		std::cout << "\tscore: " 		<< this->score		<< std::endl;
		std::cout << "\tsize: "			<< this->size		<< std::endl;
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & xmin;
		ar & ymin;
		ar & xmax;
		ar & ymax;
		ar & label;
		ar & difficult;
		ar & score;
		ar & size;
	}
};
BOOST_CLASS_VERSION(NormalizedBBox, 0);


class Annotation_s {
public:
	int instance_id;
	NormalizedBBox bbox;

	bool operator==(const Annotation_s& other) {
		return (this->instance_id == other.instance_id &&
			this->bbox == other.bbox);
		//return false;
	}

	void print() {
		std::cout << "instance_id: " << this->instance_id << std::endl;
		std::cout << "bbox: " << std::endl;
		this->bbox.print();
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & instance_id;
		ar & bbox;
	}
};
BOOST_CLASS_VERSION(Annotation_s, 0);


class AnnotationGroup {
public:
	int group_label;
	std::vector<Annotation_s> annotations;

	Annotation_s* add_annotation() {
		Annotation_s annotation;
		this->annotations.push_back(annotation);
		return &this->annotations.back();
	}

	bool operator==(const AnnotationGroup& other) {
		return (this->group_label == other.group_label &&
				std::equal(this->annotations.begin(), this->annotations.end(),
						other.annotations.begin()));
	}

	void print() {
		std::cout << "group_label: " << this->group_label << std::endl;
		for (int i = 0; i < this->annotations.size(); i++) {
			this->annotations[i].print();
		}
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & group_label;
		ar & annotations;
	}
};
BOOST_CLASS_VERSION(AnnotationGroup, 0);



class AnnotatedDatum : public Datum {
public:
	AnnotationType type;
	std::vector<AnnotationGroup> annotation_groups;
	AnnotatedDatum() {
		this->type = AnnotationType::ANNO_NONE;
		this->annotation_groups.clear();
	}

	AnnotationGroup* add_annotation_group() {
		AnnotationGroup annotation_group;
		this->annotation_groups.push_back(annotation_group);
		return &this->annotation_groups.back();
	}

	void print() {
		Datum::print();
		std::cout << "type: " << this->type << std::endl;
		for (int i = 0; i < annotation_groups.size(); i++) {
			annotation_groups[i].print();
		}
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		Datum::serialize(ar, version);
		ar & type;
		ar & annotation_groups;
	}
};
BOOST_CLASS_VERSION(AnnotatedDatum, 0);
























template <typename T>
const std::string serializeToString(T* datum) {
	std::ostringstream ofs;
	unsigned int flags = boost::archive::no_header;
	boost::archive::text_oarchive oa(ofs, flags);
	oa << (*datum);
	return ofs.str();
}

template const std::string serializeToString<Datum>(Datum* datum);
//template const std::string serializeToString<AnnotatedDatum>(AnnotatedDatum* datum);


template <typename T>
void deserializeFromString(const std::string& data, T* datum) {
	std::istringstream ifs(data);
	unsigned int flags = boost::archive::no_header;
	boost::archive::text_iarchive ia(ifs, flags);
	ia >> (*datum);
}

template void deserializeFromString<Datum>(const std::string& data, Datum* datum);
//template void deserializeFromString<AnnotatedDatum>(const std::string& data, AnnotatedDatum* datum);








#endif /* DATUM_H_ */
