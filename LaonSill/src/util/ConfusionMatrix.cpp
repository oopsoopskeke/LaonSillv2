/*
 * ConfusionMatrix.cpp
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "ConfusionMatrix.h"
#include "SysLog.h"
#include "StdOutLog.h"

std::string ConfusionMatrix::COL_SEP("\t");
std::string ConfusionMatrix::ROW_BEGIN("\t");
std::string ConfusionMatrix::ROW_END("");

ConfusionMatrix::ConfusionMatrix() {}

ConfusionMatrix::ConfusionMatrix(const int m) {
	this->_matrix.resize(m);
	for(int i = 0; i < m; i++){
		this->_matrix[i].resize(m, 0);
	}
}

ConfusionMatrix::~ConfusionMatrix() {}

int ConfusionMatrix::numRows() const {
	if (this->_matrix.empty())
		return 0;
	return (int) this->_matrix.size();
}

int ConfusionMatrix::numCols() const {
	if(this->_matrix.empty())
		return 0;
	return (int) this->_matrix[0].size();
}

void ConfusionMatrix::resize(const int m) {
	this->_matrix.resize(m);
	for(int i = 0; i < m; i++){
		this->_matrix[i].resize(m, 0);
	}
}

void ConfusionMatrix::clear() {
	for(size_t i = 0; i < this->_matrix.size(); ++i){
		std::fill(this->_matrix[i].begin(), this->_matrix[i].end(), 0);
	}
}

void ConfusionMatrix::accumulate(const int actual, const int predicted) {
	SASSERT(actual >= 0, "gt label should not be less than zero.");
	SASSERT(predicted >= 0, "prediction label should not be less than zero.");
	this->_matrix[actual][predicted] += 1;
}

void ConfusionMatrix::accumulate(const ConfusionMatrix& confusion) {
	SASSERT0(confusion._matrix.size() == this->_matrix.size());

	if(this->_matrix.empty()) return;

	SASSERT0(confusion._matrix[0].size() == this->_matrix[0].size());

	for(size_t row = 0; row < this->_matrix.size(); ++row){
		for(size_t col = 0; col < this->_matrix[row].size(); ++col){
			this->_matrix[row][col] += confusion._matrix[row][col];
		}
	}

}

void ConfusionMatrix::printCounts(const char *header) const {
	if (header == NULL) {
		STDOUT_LOG("--- confusion matrix: (actual, predicted) ---");
	} else {
		STDOUT_LOG("%s", header);
	}
	std::stringstream ss;

	for (size_t i = 0; i < this->_matrix.size(); i++) {
		ss << "\n";
		for (size_t j = 0; j < this->_matrix[i].size(); j++) {
			if (j > 0) {
				ss << " | ";
			}
			ss << this->_matrix[i][j];
		}
	}
	STDOUT_LOG("%s", ss.str().c_str());
}

void ConfusionMatrix::printRowNormalized(const char *header) const {
	if (header == NULL) {
		STDOUT_LOG("--- confusion matrix: (actual, predicted) ---");
	} else {
		STDOUT_LOG("%s", header);
	}
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		double total = rowSum(i);
		STDOUT_LOG("%s", ROW_BEGIN.c_str());
		for (size_t j = 0; j < this->_matrix[i].size(); j++) {
			if (j > 0) {
				STDOUT_LOG("%s", COL_SEP.c_str());
			}
			STDOUT_LOG("%lf", ((double)this->_matrix[i][j] / total));
		}
		STDOUT_LOG("%s", ROW_END.c_str());
	}
}

void ConfusionMatrix::printColNormalized(const char *header) const {
	std::vector<double> totals;
	for (size_t i = 0; i < this->_matrix[0].size(); i++) {
		totals.push_back(colSum(i));
	}

	if (header == NULL) {
		STDOUT_LOG("--- confusion matrix: (actual, predicted) ---");
	} else {
		STDOUT_LOG("%s", header);
	}
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		STDOUT_LOG("%s", ROW_BEGIN.c_str());
		for (size_t j = 0; j < this->_matrix[i].size(); j++) {
			if (j > 0) {
				STDOUT_LOG("%s", COL_SEP.c_str());
			}
			STDOUT_LOG("%lf", ((double)this->_matrix[i][j] / totals[j]));
		}
		STDOUT_LOG("%s", ROW_END.c_str());
	}
}

void ConfusionMatrix::printNormalized(const char *header) const {
	double total = totalSum();

	if (header == NULL) {
		STDOUT_LOG("--- confusion matrix: (actual, predicted) ---");
	} else {
		STDOUT_LOG("%s", header);
	}
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		STDOUT_LOG("%s", ROW_BEGIN.c_str());
		for (size_t j = 0; j < this->_matrix[i].size(); j++) {
			if (j > 0) STDOUT_LOG("%s", COL_SEP.c_str());
			STDOUT_LOG("%lf", ((double)this->_matrix[i][j] / total));
		}
		STDOUT_LOG("%s", ROW_END.c_str());
	}
}

void ConfusionMatrix::printPrecisionRecall(const char *header) const {
	if (header == NULL) {
		STDOUT_LOG("--- class-specific recall/precision ---");
	} else {
		STDOUT_LOG("%s", header);
	}

	// recall
	STDOUT_LOG("%s", ROW_BEGIN.c_str());
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (i > 0) {
			STDOUT_LOG("%s", COL_SEP.c_str());
		}
		double r = (this->_matrix[i].size() > i) ?
				(double)this->_matrix[i][i] / (double)rowSum(i) : 0.0;
		STDOUT_LOG("%lf", r);
	}
	STDOUT_LOG("%s", ROW_END.c_str());

	// precision
	STDOUT_LOG("%s", ROW_BEGIN.c_str());
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (i > 0) {
			STDOUT_LOG("%s", COL_SEP.c_str());
		}
		double p =
				(this->_matrix[i].size() > i) ?
						(double) this->_matrix[i][i] / (double) colSum(i) : 1.0;
		STDOUT_LOG("%lf", p);
	}
	STDOUT_LOG("%s", ROW_END.c_str());
}

void ConfusionMatrix::printF1Score(const char *header) const {
	if (header == NULL) {
		STDOUT_LOG("--- class-specific F1 score ---");
	} else {
		STDOUT_LOG("%s", header);
	}

	STDOUT_LOG("%s", ROW_BEGIN.c_str());
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (i > 0) {
			STDOUT_LOG("%s", COL_SEP.c_str());
		}
		// recall
		double r =
				(this->_matrix[i].size() > i) ?
						(double) this->_matrix[i][i] / (double) rowSum(i) : 0.0;
		// precision
		double p =
				(this->_matrix[i].size() > i) ?
						(double) this->_matrix[i][i] / (double) colSum(i) : 1.0;
		STDOUT_LOG("%lf", ((2.0 * p * r) / (p + r)));
	}
	STDOUT_LOG("%s", ROW_END.c_str());
}

void ConfusionMatrix::printJaccard(const char *header) const {
	if (header == NULL) {
		STDOUT_LOG("--- class-specific Jaccard coefficient ---");
	} else {
		STDOUT_LOG("%s", header);
	}

	STDOUT_LOG("%s", ROW_BEGIN.c_str());
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (i > 0) {
			STDOUT_LOG("%s", COL_SEP.c_str());
		}
		double p =
				(this->_matrix[i].size() > i) ?
						(double) this->_matrix[i][i]
								/ (double) (rowSum(i) + colSum(i)
										- this->_matrix[i][i]) :
						0.0;
		STDOUT_LOG("%lf", p);
	}
	STDOUT_LOG("%s", ROW_END.c_str());
}

double ConfusionMatrix::rowSum(int n) const {
	double v = 0.0;
	for (size_t i = 0; i < this->_matrix[n].size(); i++) {
		v += (double) this->_matrix[n][i];
	}
	return v;
}

double ConfusionMatrix::colSum(int m) const {
	double v = 0;
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		v += (double) this->_matrix[i][m];
	}
	return v;
}

double ConfusionMatrix::diagSum() const {
	double v = 0;
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (i >= this->_matrix[i].size())
			break;
		v += (double) this->_matrix[i][i];
	}
	return v;
}

double ConfusionMatrix::totalSum() const {
	double v = 0;
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		for (size_t j = 0; j < this->_matrix[i].size(); j++) {
			v += (double) this->_matrix[i][j];
		}
	}
	return v;
}

double ConfusionMatrix::accuracy() const {
	double total_sum = totalSum();
	double diag_sum = diagSum();

	if (total_sum == 0) {
		return 0;
	} else {
		return diag_sum / total_sum;
	}
}

double ConfusionMatrix::avgPrecision() const {
	double totalPrecision = 0.0;
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		totalPrecision +=
				(this->_matrix[i].size() > i) ?
						(double) this->_matrix[i][i] / (double) colSum(i) : 1.0;
	}

	return totalPrecision /= (double) this->_matrix.size();
}

double ConfusionMatrix::avgRecall(const bool strict) const {
	double totalRecall = 0.0;
	int numClasses = 0;
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (this->_matrix[i].size() > i) {
			const double classSize = (double) rowSum(i);
			if (classSize > 0.0) {
				totalRecall += (double) this->_matrix[i][i] / classSize;
				numClasses += 1;
			}
		}
	}

	if (strict && numClasses != (int) this->_matrix.size()) {
		SASSERT(false, "not all classes represented in avgRecall()");
	}

	if (numClasses == 0) {
		return 0;
	} else {
		return totalRecall / (double) numClasses;
	}
}

double ConfusionMatrix::avgJaccard() const {
	double totalJaccard = 0.0;
	for (size_t i = 0; i < this->_matrix.size(); i++) {
		if (this->_matrix[i].size() <= i)
			continue;
		const double intersectionSize = (double) this->_matrix[i][i];
		const double unionSize = (double) (rowSum(i) + colSum(i)
				- this->_matrix[i][i]);
		if (intersectionSize == unionSize) {
			// avoid divide by zero
			totalJaccard += 1.0;
		} else {
			totalJaccard += intersectionSize / unionSize;
		}
	}

	return totalJaccard / (double) this->_matrix.size();
}

double ConfusionMatrix::precision(int n) const {
	SASSERT0(this->_matrix.size() > (size_t) n);
	return (this->_matrix[n].size() > (size_t) n) ?
			(double) this->_matrix[n][n] / (double) colSum(n) : 1.0;
}

double ConfusionMatrix::recall(int n) const {
	SASSERT0(this->_matrix.size() > (size_t) n);
	return (this->_matrix[n].size() > (size_t) n) ?
			(double) this->_matrix[n][n] / (double) rowSum(n) : 0.0;
}

double ConfusionMatrix::jaccard(int n) const {
	SASSERT0(
			(this->_matrix.size() > (size_t) n)
					&& (this->_matrix[n].size() > (size_t) n));
	const double intersectionSize = (double) this->_matrix[n][n];
	const double unionSize = (double) (rowSum(n) + colSum(n)
			- this->_matrix[n][n]);
	return (intersectionSize == unionSize) ? 1.0 : intersectionSize / unionSize;
}

const unsigned long& ConfusionMatrix::operator()(int i, int j) const {
	return this->_matrix[i][j];
}

unsigned long& ConfusionMatrix::operator()(int i, int j) {
	return this->_matrix[i][j];
}
