
#include <stdlib.h>
#include <stdio.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/return_value_policy.hpp>

#include "Tools.h"
#include "DataReader.h"


using namespace std;


BOOST_PYTHON_MODULE(libLaonSillClient) {
	using namespace boost::python;

	// SDF Validation


	// SDF Summary
	typedef vector<int> VectorInt;
	class_<VectorInt>("VectorInt")
		.def(vector_indexing_suite<VectorInt>());

	typedef vector<long> VectorLong;
	class_<VectorLong>("VectorLong")
		.def(vector_indexing_suite<VectorLong>());

	typedef vector<float> VectorFloat;
	class_<VectorFloat>("VectorFloat")
		.def(vector_indexing_suite<VectorFloat>());

	typedef vector<string> VectorString;
	class_<VectorString>("VectorString")
		.def(vector_indexing_suite<VectorString>());

	class_<Datum>("Datum")
		.def("getImgSize",			&Datum::getImgSize)
		.def("hasLabel",			&Datum::hasLabel)
		.def("info",				&Datum::print)
		.def_readonly("channels",	&Datum::channels)
		.def_readonly("height",		&Datum::height)
		.def_readonly("width",		&Datum::width)
		.def_readonly("label",		&Datum::label)
		.def_readonly("encoded",	&Datum::encoded)
		.def_readonly("float_data",	&Datum::float_data)
		.def_readwrite("data",		&Datum::data)
	;

	enum_<AnnotationType>("AnnotationType")
		.value("ANNO_NONE", ANNO_NONE)
		.value("BBOX", BBOX)
	;

	class_<NormalizedBBox>("NormalizedBBox")
		.def("info", &NormalizedBBox::print)
		.def_readonly("xmin",		&NormalizedBBox::xmin)
		.def_readonly("ymin",		&NormalizedBBox::ymin)
		.def_readonly("xmax",		&NormalizedBBox::xmax)
		.def_readonly("ymax",		&NormalizedBBox::ymax)
		.def_readonly("label",		&NormalizedBBox::label)
		.def_readonly("difficult",	&NormalizedBBox::difficult)
		.def_readonly("score",		&NormalizedBBox::score)
		.def_readonly("size",		&NormalizedBBox::size)
	;

	class_<Annotation_s>("Annotation")
		.def("info", 					&Annotation_s::print)
		.def_readonly("instance_id",	&Annotation_s::instance_id)
		.def_readonly("bbox", 			&Annotation_s::bbox)
	;

	class_<vector<Annotation_s>>("vector_annotation")
			.def(vector_indexing_suite<vector<Annotation_s>>());

	class_<AnnotationGroup>("AnnotationGroup")
		.def("info", 					&AnnotationGroup::print)
		.def_readonly("group_label",	&AnnotationGroup::group_label)
		.def_readonly("annotations",	&AnnotationGroup::annotations)
	;
	class_<vector<AnnotationGroup>>("vector_annotationgroup")
		.def(vector_indexing_suite<vector<AnnotationGroup>>());

	class_<AnnotatedDatum, bases<Datum>>("AnnotatedDatum")
		.def("info", 						&AnnotatedDatum::print)
		.def_readonly("type", 				&AnnotatedDatum::type)
		.def_readonly("annotation_groups",	&AnnotatedDatum::annotation_groups)
	;

	typedef DataReader<class Datum> DataReaderDatum;
	class_<DataReaderDatum>("DataReaderDatum", init<const string&>())
		.def("getNumData", 		&DataReaderDatum::getNumData)
		.def("getNextData", 	&DataReaderDatum::getNextData,
				return_value_policy<manage_new_object>())
		.def("peekNextData", 	&DataReaderDatum::peekNextData,
				return_value_policy<manage_new_object>())
		.def("getHeader", 		&DataReaderDatum::getHeader,
				return_value_policy<return_by_value>())
		.def("selectDataSetByName", 	&DataReaderDatum::selectDataSetByName)
		.def("selectDataSetByIndex", 	&DataReaderDatum::selectDataSetByIndex)
	;

	typedef DataReader<class AnnotatedDatum> DataReaderAnnoDatum;
	class_<DataReaderAnnoDatum>("DataReaderAnnoDatum", init<const string&>())
		.def("getNumData",		&DataReaderAnnoDatum::getNumData)
		.def("getNextData",		&DataReaderAnnoDatum::getNextData,
				return_value_policy<manage_new_object>())
		.def("peekNextData",	&DataReaderAnnoDatum::peekNextData,
				return_value_policy<manage_new_object>())
		.def("getHeader", 		&DataReaderAnnoDatum::getHeader,
				return_value_policy<return_by_value>())
		.def("selectDataSetByName",	&DataReaderAnnoDatum::selectDataSetByName)
		.def("selectDataSetByIndex",	&DataReaderAnnoDatum::selectDataSetByIndex)
	;



	class_<BaseConvertParam>("BaseConvertParam")
		.def("info",						&BaseConvertParam::print)
		.def("validityCheck",				&BaseConvertParam::validityCheck)
		.def_readwrite("labelMapFilePath",	&BaseConvertParam::labelMapFilePath)
		.def_readwrite("outFilePath",		&BaseConvertParam::outFilePath)
		.def_readonly("resultCode",			&BaseConvertParam::resultCode)
		.def_readonly("resultMsg",			&BaseConvertParam::resultMsg)
	;

	class_<MnistDataSet>("MnistDataSet")
		.def("info",					&MnistDataSet::print)
		.def("validityCheck", 			&MnistDataSet::validityCheck)
		.def_readwrite("name", 			&MnistDataSet::name)
		.def_readwrite("imageFilePath", &MnistDataSet::imageFilePath)
		.def_readwrite("labelFilePath", &MnistDataSet::labelFilePath)
	;

	typedef vector<MnistDataSet> VectorMnistDataSet;
	class_<VectorMnistDataSet>("VectorMnistDataSet")
		.def(vector_indexing_suite<VectorMnistDataSet>());

	// SDF Building
	class_<ConvertMnistDataParam, bases<BaseConvertParam>>("ConvertMnistDataParam")
		.def("addDataSet", 				&ConvertMnistDataParam::addDataSet)
		.def("info", 					&ConvertMnistDataParam::print)
		.def_readwrite("dataSetList", 	&ConvertMnistDataParam::dataSetList)
	;


	class_<ImageSet>("ImageSet")
		.def("info",					&ImageSet::print)
		.def("validityCheck", 			&ImageSet::validityCheck)
		.def_readwrite("name",			&ImageSet::name)
		.def_readwrite("dataSetPath",	&ImageSet::dataSetPath)
	;

	typedef vector<ImageSet> VectorImageSet;
	class_<VectorImageSet>("VectorImageSet")
		.def(vector_indexing_suite<VectorImageSet>());

	class_<ConvertImageSetParam, bases<BaseConvertParam>>("ConvertImageSetParam")
		.def("addImageSet", 				&ConvertImageSetParam::addImageSet)
		.def("info", 						&ConvertImageSetParam::print)
		.def_readwrite("gray",				&ConvertImageSetParam::gray)
		.def_readwrite("shuffle",			&ConvertImageSetParam::shuffle)
		.def_readwrite("multiLabel",		&ConvertImageSetParam::multiLabel)
		.def_readwrite("channelSeparated",	&ConvertImageSetParam::channelSeparated)
		.def_readwrite("resizeWidth",		&ConvertImageSetParam::resizeWidth)
		.def_readwrite("resizeHeight",		&ConvertImageSetParam::resizeHeight)
		.def_readwrite("checkSize",			&ConvertImageSetParam::checkSize)
		.def_readwrite("encoded",			&ConvertImageSetParam::encoded)
		.def_readwrite("encodeType",		&ConvertImageSetParam::encodeType)
		.def_readwrite("basePath",			&ConvertImageSetParam::basePath)
		.def_readwrite("imageSetList",		&ConvertImageSetParam::imageSetList)
	;
	class_<ConvertAnnoSetParam, bases<ConvertImageSetParam>>("ConvertAnnoSetParam")
		.def("info", 						&ConvertAnnoSetParam::print)
		.def_readwrite("annoType",			&ConvertAnnoSetParam::annoType)
		.def_readwrite("labelType",			&ConvertAnnoSetParam::labelType)
		.def_readwrite("checkLabel",		&ConvertAnnoSetParam::checkLabel)
		.def_readwrite("minDim",			&ConvertAnnoSetParam::minDim)
		.def_readwrite("maxDim",			&ConvertAnnoSetParam::maxDim)
	;

	def("ConvertMnistData", convertMnistData);
	def("ConvertImageSet", convertImageSet);
	def("ConvertAnnoSet", convertAnnoSet);

	// ETC Tools
	def("Denormalize", denormalize);
	def("ComputeImageMean", computeImageMean);


	class_<LabelItem>("LabelItem")
		.def("info", 					&LabelItem::print)
		.def_readwrite("name", 			&LabelItem::name)
		.def_readwrite("label",			&LabelItem::label)
		.def_readwrite("displayName",	&LabelItem::displayName)
		.def_readwrite("color", 		&LabelItem::color)
	;
	typedef vector<LabelItem> VectorLabelItem;
	class_<VectorLabelItem>("VectorLabelItem")
		.def(vector_indexing_suite<VectorLabelItem>());

	class_<SDFHeader>("SDFHeader")
		.def("info", 					&SDFHeader::print)
		.def_readwrite("numSets", 		&SDFHeader::numSets)
		.def_readwrite("names",			&SDFHeader::names)
		.def_readwrite("setSizes",		&SDFHeader::setSizes)
		.def_readwrite("setStartPos",	&SDFHeader::setStartPos)
		.def_readwrite("labelItemList",	&SDFHeader::labelItemList)
		.def_readonly("type", 			&SDFHeader::type)
		.def_readonly("uniform",		&SDFHeader::uniform)
		.def_readonly("channels",		&SDFHeader::channels)
		.def_readonly("minHeight",		&SDFHeader::minHeight)
		.def_readonly("minWidth",		&SDFHeader::minWidth)
		.def_readonly("maxHeight",		&SDFHeader::maxHeight)
		.def_readonly("maxWidth",		&SDFHeader::maxWidth)
		.def_readonly("numClasses",		&SDFHeader::numClasses)
		.def_readonly("size",			&SDFHeader::size)
		.def_readonly("version",		&SDFHeader::version)
	;

	def("RetrieveSDFHeader", SDF::retrieveSDFHeader);


	enum_<Mode>("Mode")
		.value("READ", READ)
		.value("NEW", NEW)
	;
	class_<SDF>("SDF", init<const string&, const Mode>())
		.def("open",			&SDF::open)
		.def("close",			&SDF::close)
		.def("put",				&SDF::put)
		.def("commit",			&SDF::commit)
		//.def("getNextValue",	&SDF::getNextValue, return_value_policy<manage_new_object>())
	;




}



















