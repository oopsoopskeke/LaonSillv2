/**
 * @file Update.cpp
 * @date 2017-05-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Update.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "Worker.h"
#include "Network.h"
#include "MathFunctions.h"
#include "StdOutLog.h"

template<typename Dtype>
void Update<Dtype>::updateParam(UpdateContext context, Data<Dtype>* dataHistory,
    Data<Dtype>* dataHistory2, Data<Dtype>* data) {

    if (!SLPROP_BASE(updateGrad))
        return;

    int paramSize = context.paramSize;
    Dtype regScale = (Dtype)context.regScale;
    Dtype learnScale = (Dtype)context.learnScale;
    Dtype epsilon = (Dtype)context.epsilon;
    Dtype decayRate = (Dtype)context.decayRate;
    Dtype beta1 = (Dtype)context.beta1;
    Dtype beta2 = (Dtype)context.beta2;
    Dtype decayedBeta1 = (Dtype)context.decayedBeta1;
    Dtype decayedBeta2 = (Dtype)context.decayedBeta2;

	const uint32_t batches = SNPROP(batchSize);
	const Dtype normScale = 1.0/batches;
	const Dtype momentum = SNPROP(momentum);
	const Dtype negativeOne = -1.0;
    const Dtype negativeLearnScale = (-1.0) * learnScale;

#if 0
    if (!Worker<Dtype>::isSingle())
        data->mutable_host_grad();
#endif

	Dtype* d_paramGrad = data->mutable_device_grad();
	Dtype* d_paramData = data->mutable_device_data();
    Dtype* d_paramHistoryData = NULL;
    Dtype* d_paramHistoryData2 = NULL;

    Optimizer opt = (Optimizer)SNPROP(optimizer);
    int paramHistoryDataCount = getParamHistoryDataCount(opt);
    if (paramHistoryDataCount == 2) {
	    d_paramHistoryData = dataHistory->mutable_device_data();
	    d_paramHistoryData2 = dataHistory2->mutable_device_data();
    } else if (paramHistoryDataCount == 1) {
	    d_paramHistoryData = dataHistory->mutable_device_data();
    } else {
        SASSUME0(paramHistoryDataCount == 0);
    }

    // Gradients Clipping ***
    if (SNPROP(doClipGradients))
        Update<Dtype>::clipGradients(paramSize, d_paramGrad);

    // apply optimizer
    if (opt == Optimizer::Momentum) {
        /****
         * Momentum Alogorithm
         *
         * v = mu * v - learning_rate * dx
         * x += v
         *
         */

#if 0
    	std::cout << "paramSize: " << paramSize << ", regScale: " << regScale << std::endl;
    	std::cout << "learnScale: " << learnScale << ", momentum: " << momentum << std::endl;
    	Data<Dtype>::printConfig = 1;
    	SyncMem<Dtype>::printConfig = 1;
    	data->print_data({}, false);
    	data->print_grad({}, false);
#endif
    	soooa_gpu_axpy(paramSize, regScale, d_paramData, d_paramGrad);
    	//data->print_grad({}, false);

    	//dataHistory->print_data({}, false);
		soooa_gpu_axpby(paramSize, learnScale, d_paramGrad, momentum,
				d_paramHistoryData);
		//dataHistory->print_data({}, false);

		soooa_copy(paramSize, d_paramHistoryData, d_paramGrad);
		//data->print_grad({}, false);

		// update
		soooa_gpu_axpy(paramSize, negativeOne, d_paramGrad, d_paramData);
		//data->print_data({}, false);


		//Data<Dtype>::printConfig = 0;
		//SyncMem<Dtype>::printConfig = 0;

    } else if (opt == Optimizer::Vanilla) {
        /****
         * Vanilla Alogorithm
         *
         * x += -learning_rate * dx
         *
         */
    	checkCUBLAS(cublasSscal(Cuda::cublasHandle, paramSize,
            &learnScale, d_paramGrad, 1));				//
    	checkCUBLAS(cublasSaxpy(Cuda::cublasHandle, paramSize,
            &negativeOne, d_paramGrad, 1, d_paramData, 1));		// update
    } else if (opt == Optimizer::Nesterov) {
        /****
         * Nesterov Alogorithm
         *
         * v_prev = v # back this up
         * v = mu * v - learning_rate * dx # velocity update stays the same
         * x += -mu * v_prev + (1 + mu) * v # position update changes form
         *
         */
	    Update<Dtype>::doNesterov(paramSize, d_paramGrad,
            d_paramHistoryData, d_paramHistoryData2, d_paramData, momentum, learnScale);
    } else if (opt == Optimizer::Adagrad) {
        /****
         * Adagrad Alogorithm
         *
         * cache += dx**2
         * x += -learning_rate * dx / (sqrt(cache) + eps)
         *
         */
	    Update<Dtype>::doAdagrad(paramSize, d_paramGrad,
            d_paramHistoryData, d_paramData, learnScale, epsilon);

    } else if (opt == Optimizer::RMSprop) {
        /****
         * RMSprop
         *
         * cache = decay_rate * cache + (1 - decay_rate) * dx**2
         * x += - learning_rate * dx / (sqrt(cache) + eps)
         *
         */
	    Update<Dtype>::doRMSprop(paramSize, d_paramGrad,
            d_paramHistoryData, d_paramData, learnScale, epsilon, decayRate);

    } else if (opt == Optimizer::Adam) {
        /****
         * Adam
         *
         * m = beta1 * m + (1 - beta1) * dx
         * v = beta2 * v + (1 - beta2) * (dx**2)
         * x += -learning_rate * m / (sqrt(v) + eps)
         *
         */
        Update<Dtype>::doAdam(paramSize, d_paramGrad, d_paramHistoryData,
            d_paramHistoryData2, d_paramData, learnScale, epsilon, beta1, beta2,
            decayedBeta1, decayedBeta2);
    } else if (opt == Optimizer::Adadelta) {
        /****
         * Adadelta
         *
         * e1 = momentum * e1 + (1 - momentum) * (dx ** 2)
         * delta = dx * sqrt((e2 + epsilon) / (e1 + epsilon))
         * e2 = moemntum * e2 + (1 - momentum) * (delta ** 2)
         * x += -learning_rate * delta
         *
         */
        Update<Dtype>::doAdadelta(paramSize, d_paramGrad, d_paramHistoryData,
            d_paramHistoryData2, d_paramData, momentum, learnScale, epsilon);
    } else {
        SASSERT(false, "invalid optimizer. optimizer=%d", (int)opt);
    }
}

/**
 * learning rate policy (from CAFFE definition)
 *    - fixed: always return base_lr.
 *    - step: return base_lr * gamma ^ (floor(iter / step))
 *    - exp: return base_lr * gamma ^ iter
 *    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
 *    - multistep: similar to step but it allows non uniform steps defined by
 *      stepvalue
 *    - poly: the effective learning rate follows a polynomial decay, to be
 *      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
 *    - sigmoid: the effective learning rate follows a sigmod decay
 *      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
 */
template <typename Dtype>
float Update<Dtype>::calcLearningRate() {
	float rate;
	switch (SNPROP(lrPolicy)) {
        case Fixed: {
            rate = SNPROP(baseLearningRate);
        }
            break;
        case Step: {
            int currentStep = SNPROP(iterations) / SNPROP(stepSize);
            rate = SNPROP(baseLearningRate) * pow(SNPROP(gamma), currentStep);

            if (SNPROP(rate) != rate) {
            	STDOUT_LOG("Learning rate updated: : %f", rate);
            }
            if (SNPROP(rate) < 0.0f || SNPROP(rate) != rate) {
                SNPROP(rate) = rate;
            }
        }
            break;
        case Poly: {
        	if (SNPROP(miniBatch) > 0) {
        		rate = SNPROP(baseLearningRate) *
        				pow(1.0 - (float)SNPROP(iterations) /
        						((float)SNPROP(epochs) * (float)SNPROP(miniBatch)), SNPROP(power));
        	} else if (SNPROP(maxIterations) > 0){
				rate = SNPROP(baseLearningRate) *
					pow(1.0 - ((float)(SNPROP(iterations)-1) / (float)SNPROP(maxIterations)), 
                            SNPROP(power));
        	} else {
        		SASSERT(false, "one of miniBatch and maxIterations should be larger than 0");
        	}
        }
            break;
        case Multistep: {
        	const std::vector<int>& stepValue = SNPROP(stepValue);
        	const int iterations = SNPROP(iterations);
        	if (SNPROP(currentStep) < stepValue.size() &&
        			iterations >= stepValue[SNPROP(currentStep)]) {
        		SNPROP(currentStep) = SNPROP(currentStep) + 1;
        		STDOUT_LOG("MultiStep Status: Iteration %d, step = %u", iterations,
        				SNPROP(currentStep));
        	}

            if (SNPROP(stepScale).size() == 0)
        	    rate = SNPROP(baseLearningRate) * pow(SNPROP(gamma), SNPROP(currentStep));
            else
                rate = SNPROP(baseLearningRate) * SNPROP(stepScale)[SNPROP(currentStep)];
        }
        	break;

        case Inv: {
        	float baseLearningRate = SNPROP(baseLearningRate);
        	float gamma = SNPROP(gamma);
        	int iterations = SNPROP(iterations);
        	float power = SNPROP(power);
        	rate = baseLearningRate * pow(Dtype(1) + gamma * iterations, - power);
        }
        	break;

        default: {
            SASSERT(false, "Unsupported LRPolicy");
        }
	}

	return rate;
}

template<typename Dtype>
UpdateContext Update<Dtype>::makeContext(int paramSize, float regScale, float learnScale) {
    UpdateContext context;
    context.paramSize = paramSize;
    context.regScale = regScale;
    context.learnScale = learnScale;
    context.epsilon = SNPROP(epsilon);
    context.decayRate = SNPROP(decayRate);
    context.beta1 = SNPROP(beta1);
    context.beta2 = SNPROP(beta2);
    context.decayedBeta1 = SLPROP_LEARN(decayedBeta1);
    context.decayedBeta2 = SLPROP_LEARN(decayedBeta2);

    return context;
}

template<typename Dtype>
int Update<Dtype>::getParamHistoryDataCount(Optimizer opt) {

    if (opt == Optimizer::Vanilla)
        return 0;

    if (opt == Optimizer::Momentum || 
        opt == Optimizer::Adagrad || 
        opt == Optimizer::RMSprop) {
        return 1;
    }

    if (opt == Optimizer::Nesterov || 
        opt == Optimizer::Adam || 
        opt == Optimizer::Adadelta) {
        return 2;
    }

    SASSUME(false, "wrong optimizer. optimizer=%d", (int)opt);
}

// Gradients Clipping ***
template <typename Dtype>
void Update<Dtype>::clipGradients(const int size, Dtype* dx) {
    Dtype clipGrad = SNPROP(clipGradientsLevel);
    if (clipGrad > 0.0) {
        Dtype l2normDiff;
        checkCUBLAS(cublasSnrm2(Cuda::cublasHandle, size, dx, 1, &l2normDiff));
        if (l2normDiff > clipGrad) {

        Dtype scaleFactor = clipGrad / l2normDiff;
        checkCUBLAS(cublasSscal(Cuda::cublasHandle, size, &scaleFactor, dx, 1));
        //std::cout << "Gradient clipping: scaling down gradients" << std::endl;
        //std::cout << "(L2 norm " << l2normDiff << " > " << clipGrad << ")" << std::endl;
        //std::cout << "by scale factor " << scaleFactor << std::endl;
        }
    }
}

template class Update<float>;
