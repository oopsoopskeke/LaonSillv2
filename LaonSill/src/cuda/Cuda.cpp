/*
 * Cuda.cpp
 *
 *  Created on: 2016. 6. 16.
 *      Author: jhkim
 */

#include "Cuda.h"
#include "Util.h"
#include "ColdLog.h"
#include "SysLog.h"

using namespace std;

//int Cuda::gpuid = 0;
int Cuda::gpuCount;
const float Cuda::alpha = 1.0f;
const float Cuda::beta = 0.0f;
const float Cuda::negativeOne = -1.0f;

vector<int> Cuda::availableGPU; // peer access가 가능한 GPU만 지원함.

thread_local cudnnHandle_t Cuda::cudnnHandle;
thread_local cublasHandle_t Cuda::cublasHandle;


Cuda::Cuda() {}
Cuda::~Cuda() {}

void Cuda::create(int usingGPUCount) {
	int devGPUCount;	// 머신에서 제공하는 GPU 개수
	int i, j;

	checkCudaErrors(cudaGetDeviceCount(&devGPUCount));
	if(devGPUCount == 0) {
        SYS_LOG("ERROR: There is zero GPUs on this machine");
		exit(1);
	}

	if(usingGPUCount > devGPUCount) {
        SYS_LOG("ERROR: Invalid GPU count %d (There are %d GPUs on this machine)",
            usingGPUCount, devGPUCount);
		exit(1);
	}

	if(usingGPUCount <= 0) {
		Cuda::gpuCount = devGPUCount;
	} else {
		Cuda::gpuCount = usingGPUCount;
	}

    // gpu가 하나이면 peer access 확인을 하지 않는다.
    if (Cuda::gpuCount == 1) {
        Cuda::availableGPU.push_back(0);
        SYS_LOG("This machine uses 1 GPU");
        return;
    }

    // gpu가 여러개이면 peer access 확인을 한다.
    for (i = 0; i < Cuda::gpuCount; i++) {
        bool canAccessAny = false;  // 하나라도 접근 가능하면 true, 
                                    // 해당 기능이 있더라도 2개이상 peer access 기능이 없으면
                                    // 기능이 없는것과 동일함.
        int canAccess;
        for (j = 0; j < Cuda::gpuCount; j++) {
            if (i == j)
                continue;

            checkCudaErrors(cudaDeviceCanAccessPeer(&canAccess, i , j));
            if (canAccess) {
                canAccessAny = true;
                break;
            }
        }

        if (canAccessAny) {
            Cuda::availableGPU.push_back(i);
            SYS_LOG("GPU #%d is added", i);
        }
    }

    if (Cuda::availableGPU.size() < 1) {
        SYS_LOG("ERROR: No peer-accessible GPU on this machines.");
        exit(0);
    }

    Cuda::gpuCount = Cuda::availableGPU.size();

    for (i = 0; i < Cuda::availableGPU.size(); i++) {
        checkCudaErrors(cudaSetDevice(Cuda::availableGPU[i]));
        for (j = 0; j < Cuda::availableGPU.size(); j++) {
            if (i == j)
                continue;

            checkCudaErrors(cudaDeviceEnablePeerAccess(Cuda::availableGPU[j], 0));
        }
    }

    SYS_LOG("This machine uses %d GPUs.", Cuda::gpuCount);
}

void Cuda::destroy() {
	return;
}

void Cuda::refresh() {
	/* deprecated function */
	return;
}


