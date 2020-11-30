/*
 * DataReader.h
 *
 *  Created on: Jun 30, 2017
 *      Author: jkim
 */

#ifndef DATAREADER_H_
#define DATAREADER_H_

#include <string>
#include <queue>

#include "SDF.h"

template <typename T>
class DataReader {
public:
	DataReader(const std::string& source);
	DataReader(const DataReader<T>& dataReader);
	virtual ~DataReader();

	T* getNextData();
	T* peekNextData();
    void fillNextData(T* data);
    void selectDataSetByName(const std::string& dataSet);
    void selectDataSetByIndex(const int dataSetIdx);

	int getNumData();
	SDFHeader getHeader();

	std::string source;

private:
	SDF db;
	//int numData;

	//std::queue<T*> data_queue;

    /***************************************************************************
     * callback functions - will be registered by Input Data Provider module
     * *************************************************************************/
public:
    static void allocElem(void** elemPtr);
    static void deallocElem(void* elemPtr);
    static void fillElem(void* reader, void* elemPtr);
};

#endif /* DATAREADER_H_ */
