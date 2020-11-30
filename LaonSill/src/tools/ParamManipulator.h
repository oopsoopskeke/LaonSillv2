/*
 * ParamManipulator.h
 *
 *  Created on: Jun 27, 2017
 *      Author: jkim
 */

#ifndef PARAMMANIPULATOR_H_
#define PARAMMANIPULATOR_H_

#include <map>
#include <vector>

#include "Data.h"


template <typename Dtype>
class ParamManipulator {
public:
	ParamManipulator(const std::string& oldParamPath, const std::string& newParamPath);
	virtual ~ParamManipulator();

	void printParamList();

	void changeParamNames(const std::vector<std::pair<std::string, std::string>>& namePairList);
	void changeParamName(const std::string& oldParamName, const std::string& newParamName);
	void denormalizeParams(const std::vector<std::string>& paramNames,
			const std::vector<float>& means, const std::vector<float>& stds);

	void save();

private:
	void loadParam();
	Data<Dtype>* findParam(const std::string& paramName);


private:
	std::vector<Data<Dtype>*> dataList;
	std::map<std::string, Data<Dtype>*> dataMap;

	std::string oldParamPath;
	std::string newParamPath;


};

#endif /* DATAMANIPULATOR_H_ */
