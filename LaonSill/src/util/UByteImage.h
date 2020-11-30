/**
 * @file UbyteImage.h
 * @date 2016/7/14
 * @author jhkim
 * @brief Mnist의 데이터를 기준, 데이터셋을 읽어들일 때 사용하는 자료구조와 함수들을 포함한다.
 * @details
 */

#ifndef UBYTEIMAGE_H_
#define UBYTEIMAGE_H_

#include "common.h"

#define UBYTE_IMAGE_MAGIC 2051
#define UBYTE_LABEL_MAGIC 2049

#ifdef _MSC_VER
	#define bswap(x) _byteswap_ulong(x)
#else
	#define bswap(x) __builtin_bswap32(x)
#endif


/**
 * @brief 데이터셋 파일 헤더 구조체
 */
struct UByteImageDataset {
	uint32_t magic;			///< 매직 넘버 (UBYTE_IMAGE_MAGIC).
	uint32_t length;		///< 데이터셋 파일에 들어있는 이미지의 수
	uint32_t height;		///< 각 이미지의 높이값
	uint32_t width;			///< 각 이미지의 너비값
	uint32_t channel;		///< 각 이미지의 채널값
	/**
	 * @details 헤더의 각 필드별로 swap하여 저장한다.
	 */
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
		height = bswap(height);
		width = bswap(width);
		channel = bswap(channel);
	}
};
/**
 * @brief 데이터셋 정답 파일 헤더 구조체
 */
struct UByteLabelDataset {
	uint32_t magic;			///< 매직 넘버 (UBYTE_LABEL_MAGIC).
	uint32_t length;		///< 데이터셋 파일에 들어있는 정답의 수
	/**
	 * @details 헤더의 각 필드별로 swap하여 저장한다.
	 */
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
	}
};




#endif /* UBYTEIMAGE_H_ */
