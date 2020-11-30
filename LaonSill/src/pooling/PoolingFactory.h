/**
 * @file PoolingFactory.h
 * @date 2016/6/7
 * @author jhkim
 * @brief
 * @details
 */

#ifndef POOLING_POOLINGFACTORY_H_
#define POOLING_POOLINGFACTORY_H_

#include "common.h"
#include "AvgPooling.h"
#include "MaxPooling.h"
#include "MemoryMgmt.h"

/**
 * @brief 주어진 풀링 타입에 따라 풀링 객체를 생성하여 반환
 * @details 주어진 풀링 타입에 따라 풀링 객체를 생성하여 반환하고
 *          사용이 완료된 풀링 객체를 소멸시키는 역할을 함.
 * @todo (객체를 생성한 곳에서 삭제한다는 원칙에 따라 만들었으나 수정이 필요)
 */
template <typename Dtype>
class PoolingFactory {
public:
	PoolingFactory() {}
	virtual ~PoolingFactory() {}

	/**
	 * @details 주어진 풀링 타입에 따라 풀링 객체를 생성하여 반환.
	 * @param poolingType 생성하고자 하는 풀링 객체의 타입.
	 * @param pool_d 풀링 연산 관련 파라미터 구조체
	 * @return 생성한 풀링 객체.
	 */
	static Pooling<Dtype>* create(PoolingType poolingType, pool_dim pool_d) {
		//Pooling<Dtype>* ptr = NULL;
		switch(poolingType) {
		case PoolingType::Max:
			//SNEW(ptr, MaxPooling<Dtype>, pool_d);
			//SASSUME0(ptr != NULL);
			//return ptr;
			return new MaxPooling<Dtype>(pool_d);
		case PoolingType::Avg:
			//SNEW(ptr, AvgPooling<Dtype>, pool_d);
			//SASSUME0(ptr != NULL);
			//return ptr;
			return new AvgPooling<Dtype>(pool_d);
		default:
			return NULL;
		}
	}

	/**
	 * @details PoolingFactory에서 생성한 풀링 객체를 소멸.
	 * @param pooling_fn 풀링 객체에 대한 포인터 참조자.
	 */
	static void destroy(Pooling<Dtype>*& pooling_fn) {
		if(pooling_fn) {
			//SDELETE(pooling_fn);
            delete pooling_fn;
			pooling_fn = NULL;
		}
	}
    
};

template class PoolingFactory<float>;

#endif /* POOLING_POOLINGFACTORY_H_ */
