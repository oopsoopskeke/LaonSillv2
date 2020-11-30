/**
 * @file Pooling.h
 * @date 2016/5/16
 * @author jhkim
 * @brief
 * @details
 */

#ifndef POOLING_POOLING_H_
#define POOLING_POOLING_H_

#include "common.h"
#include "EnumDef.h"


/**
 * @brief Pooling 구현 클래스의 베이스 추상 클래스.
 * @details	Pooling 클래스를 상속받아 풀링을 구현하는 클래스를 생성할 수 있음.
 */
template <typename Dtype>
class Pooling {
public:
	/**
	 * @details Pooling 기본 생성자
	 */
	Pooling() {}
	/**
	 * @details Pooling 소멸자
	 */
	virtual ~Pooling() {}



	/**
	 * @details 풀링 타입을 조회한다.
	 * @return 풀링 타입
	 */
	PoolingType getType() const { return this->type; }
	cudnnPoolingDescriptor_t getPoolDesc() const { return poolDesc; }

	/**
	 * @details 주어진 입력에 대해 풀링한다.
	 * @param xDesc 입력값 x의 데이터 구성을 설명하는 cudnnTensorDescriptor 포인터
	 * @param x 입력값 장치 메모리 포인터
	 * @param yDesc 출력값 y의 데이터 구성을 설명하는 cudnnTensorDescriptor 포인터
	 * @param y 출력값 장치 메모리 포인터
	 */
	virtual void forward(const cudnnTensorDescriptor_t xDesc, const Dtype* x,
			const cudnnTensorDescriptor_t yDesc, Dtype* y)=0;
	/**
	 * @details 입력 x에 관한 gradient를 구한다.
	 * @param yDesc 출력 y의 데이터 구성을 설명하는 cudnnTensorDescriptor 포인터
	 * @param y 출력 데이터 장치 메모리 포인터
	 * @param dy 출력 그레디언트 장치 메모리 포인터
	 * @param xDesc 입력 x의 데이터 구성을 설명하는 cudnnTensorDescriptor 포인터
	 * @param x 입력 데이터 장치 메모리 포인터
	 * @param dx 입력 그레디언트 장치 메모리 포인터
	 */
	virtual void backward(const cudnnTensorDescriptor_t yDesc, const Dtype* y, 
        const Dtype* dy, const cudnnTensorDescriptor_t xDesc, const Dtype* x, Dtype* dx)=0;


protected:
	PoolingType type;									///< 풀링 타입
	cudnnPoolingDescriptor_t poolDesc;			///< cudnn 풀링 연산 정보 구조체

};

template class Pooling<float>;

#endif /* POOLING_POOLING_H_ */
