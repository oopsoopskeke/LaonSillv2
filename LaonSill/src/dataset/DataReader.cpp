/*
 * DataReader.cpp
 *
 *  Created on: Jun 30, 2017
 *      Author: jkim
 */

#include "DataReader.h"
#include "Datum.h"
#include "SysLog.h"
#include "MemoryMgmt.h"
#include "Param.h"

using namespace std;

template <typename T>
DataReader<T>::DataReader(const string& source)
: source(source), db(source, Mode::READ) {
	this->db.open();

	//string value = this->db.getNextValue();
	//this->numData = atoi(value.c_str());
	//SDFHeader header = this->db.getHeader();


}

template <typename T>
DataReader<T>::DataReader(const DataReader<T>& dataReader)
: DataReader(dataReader.source) {}

template <typename T>
DataReader<T>::~DataReader() {
	this->db.close();
}




template <typename T>
T* DataReader<T>::getNextData() {
    if (SPARAM(DATAREADER_USE_PEEK_INSTEADOF_GET))
        return peekNextData();

	string value = this->db.getNextValue();
	T* datum = NULL;
	SNEW(datum, T);
	SASSUME0(datum != NULL);
	deserializeFromString(value, datum);
	return datum;
}

template <typename T>
T* DataReader<T>::peekNextData() {
	long currentPos = this->db.getCurrentPos();
	T* datum = getNextData();
	this->db.setCurrentPos(currentPos);

	return datum;
}



template <typename T>
void DataReader<T>::fillNextData(T* data) {
    //SASSUME0(this->data_queue.size() == 0);
    string value = this->db.getNextValue();
    deserializeFromString(value, data);
}


template <typename T>
void DataReader<T>::selectDataSetByName(const string& dataSet) {
	this->db.selectDataSet(dataSet);
}

template <typename T>
void DataReader<T>::selectDataSetByIndex(const int dataSetIdx) {
	this->db.selectDataSet(dataSetIdx);
}








template <typename T>
int DataReader<T>::getNumData() {
	//return this->numData;
	return this->db.getHeader().setSizes[0];
}

template <typename T>
SDFHeader DataReader<T>::getHeader() {
	return this->db.getHeader();
}

/**************************************************************************************
 * Callback functions
 * ***********************************************************************************/

template <typename T>
void DataReader<T>::allocElem(void** elemPtr) {
	T* ptr = NULL;
	SNEW(ptr, T);
	SASSUME0(ptr != NULL);
    (*elemPtr) = (void*)ptr;
}

template <typename T>
void DataReader<T>::deallocElem(void* elemPtr) {

    SDELETE((T*)elemPtr);
}

template <typename T>
void DataReader<T>::fillElem(void* reader, void* elemPtr) {
	DataReader<T>* dataReader = (DataReader<T>*)reader;
    dataReader->fillNextData((T*)elemPtr);
}

template class DataReader<Datum>;
template class DataReader<AnnotatedDatum>;

