/*
 * LayerTestInterface.h
 *
 *  Created on: Feb 13, 2017
 *      Author: jkim
 */

#ifndef LAYERTESTINTERFACE_H_
#define LAYERTESTINTERFACE_H_

#include "TestUtil.h"

template <typename Dtype>
class LayerTestInterface {
public:
	LayerTestInterface() {}
	virtual ~LayerTestInterface() {}

	static void globalSetUp(const int gpuid) {
		setUpCuda(gpuid);
	}

	static void globalCleanUp() {
		cleanUpCuda();
	}

	virtual void setUp() = 0;
	virtual void cleanUp() = 0;
	virtual void forwardTest() = 0;
	virtual void backwardTest() = 0;
};



#endif /* LAYERTESTINTERFACE_H_ */
