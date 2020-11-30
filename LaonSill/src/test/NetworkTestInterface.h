/*
 * NetworkTestInterface.h
 *
 *  Created on: Feb 13, 2017
 *      Author: jkim
 */

#ifndef NETWORKTESTINTERFACE_H_
#define NETWORKTESTINTERFACE_H_

#include "TestUtil.h"

template <typename Dtype>
class NetworkTestInterface {
public:
	NetworkTestInterface() {}
	virtual ~NetworkTestInterface() {}

	static void globalSetUp(const int gpuid) {
		setUpCuda(gpuid);
	}

	static void globalCleanUp() {
		cleanUpCuda();
	}

	virtual void setUp() = 0;
	virtual void cleanUp() = 0;

	virtual void updateTest() = 0;
};



#endif /* NETWORKTESTINTERFACE_H_ */
