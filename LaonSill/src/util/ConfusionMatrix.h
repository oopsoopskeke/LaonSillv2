/*
 * ConfusionMatrix.h
 *
 *  Created on: Aug 7, 2017
 *      Author: jkim
 */

#ifndef CONFUSIONMATRIX_H_
#define CONFUSIONMATRIX_H_

#include <vector>
#include <string>

class ConfusionMatrix {
public:
	// create a m-by-m confusion matrix
	ConfusionMatrix();
	explicit ConfusionMatrix(const int m);
	virtual ~ConfusionMatrix();

	void accumulate(const int actual, const int predicted);
	void accumulate(const ConfusionMatrix& conf);

	int numRows() const;
	int numCols() const;

	void resize(const int m);
	void clear();

	void printCounts(const char *header = NULL) const;
	void printRowNormalized(const char *header = NULL) const;
	void printColNormalized(const char *header = NULL) const;
	void printNormalized(const char *header = NULL) const;
	void printPrecisionRecall(const char *header = NULL) const;
	void printF1Score(const char *header = NULL) const;
	void printJaccard(const char *header = NULL) const;

	double rowSum(int n) const;
	double colSum(int m) const;
	double diagSum() const;
	double totalSum() const;
	double accuracy() const;
	double avgPrecision() const;
	double avgRecall(const bool strict = true) const;
	double avgJaccard() const;

	double precision(int n) const;
	double recall(int n) const;
	double jaccard(int n) const;

	const unsigned long& operator()(int x, int y) const;
	unsigned long& operator()(int x, int y);

protected:
	// use unsigned long: be caureful of overflow for large-scale dataset
	std::vector< std::vector<unsigned long> > _matrix;

public:
	static std::string COL_SEP;   // string for separating columns when printing
	static std::string ROW_BEGIN; // string for starting a row when printing
	static std::string ROW_END;   // string for ending a row when printing

};

#endif /* CONFUSIONMATRIX_H_ */
