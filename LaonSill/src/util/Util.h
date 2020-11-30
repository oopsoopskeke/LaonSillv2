/**
 * @file Util.h
 * @date 2016/4/20
 * @author jhkim
 * @brief 다양한 유틸리티 함수, 매크로, 정의등을 선언
 * @details
 * @todo 기능을 세분화하여 파일로 나누어 정리할 필요가 있다.
 */

#ifndef UTIL_H_
#define UTIL_H_

//#include "layer/LayerConfig.h"
#include <string>
#include <fstream>
#include <fstream>

#include "Cuda.h"
#include "common.h"
#include "SysLog.h"
#include "MemoryMgmt.h"

#define C_MEM(cb, r, c, s) cb.mem[r + c*cb.n_rows + s*cb.n_elem_slice]
#define C_MEMPTR(cb, r, c, s) cb.memptr()[r + c*cb.n_rows + s*cb.n_elem_slice]
#define M_MEM(m, r, c) m.mem[r + c*m.n_rows]
#define M_MEMPTR(m, r, c) m.memptr()[r + c*m.n_rows]

#ifndef GPU_MODE
typedef fvec rvec;
typedef fmat rmat;
typedef fcube rcube;
#endif

typedef unsigned int UINT;
typedef float DATATYPE;

/**
 * @brief 각종 유틸리티 함수들을 정적으로 포함하는 클래스
 * @details
 */
class Util {
public:
	Util() {}
	virtual ~Util() {}

	template<class T>
	static void clearVector(std::vector<T*>& vec) {
		const uint32_t vecSize = vec.size();
		for(uint32_t i = 0; i < vecSize; i++) {
            if (vec[i] != NULL)
			    SDELETE(vec[i]);
		}
		vec.clear();
	}

	/**
	 * @details 최대, 최소 구간의 임의의 수를 생성한다.
	 * @param min 생성할 난수의 최소값
	 * @param max 생성할 난수의 최대값
	 * @return 생성한 난수
	 */
	static int random(int min, int max);
	/**
	 * @details 4개의 연속된 byte를 하나의 정수로 pack한다.
	 * @param buffer 4개의 연속된 byte를 저장하는 배열의 포인터
	 * @return pack한 정수값
	 */
	static int pack4BytesToInt(unsigned char *buffer);

	static uint32_t vecCountByAxis(const std::vector<uint32_t>& vec, const uint32_t axis=0);

	static void saveStringToFstream(std::ofstream& ofs, const std::string& str);
	static void loadStringFromFstream(std::ifstream& ifs, std::string& str);

#ifndef GPU_MODE
	/**
	 * @details rvec 타입의 벡터를 메타정보와 함께 설정된 출력 스트림에 출력한다.
	 * @param vector 출력할 벡터
	 * @param name 출력에 사용할 레이블
	 */
	static void printVec(const rvec &vector, std::string name);
	/**
	 * @details rmat 타입의 행렬을 메타정보와 함께 설정된 출력 스트림에 출력한다.
	 * @param matrix 출력할 행렬
	 * @param name 출력에 사용할 레이블
	 */
	static void printMat(const rmat &matrix, std::string name);
	/**
	 * @details rcube 타입의 3차원 행렬을 메타정보와 함께 설정된 출력 스트림에 출력한다.
	 * @param c 출력할 3차원 행렬
	 * @param name 출력에 사용할 레이블
	 */
	static void printCube(const rcube &c, std::string name);
	/**
	 * @details ucube 타입의 3차원 행렬을 메타정보와 함께 설정된 출력 스트림에 출력한다.
	 * @param c 출력할 3차원 행렬
	 * @param name 출력에 사용할 레이블
	 */
	static void printUCube(const ucube &c, std::string name);
#endif
	/**
	 * @details 호스트의 메모리를 지정된 데이터 구조로 메타정보와 함께 설정된 출력 스트림에
     *         출력한다.
	 * @param data 출력할 데이터 호스트 배열 포인터
	 * @param rows 데이터 배열을 해석할 row값
	 * @param cols 데이터 배열을 해석할 col값
	 * @param channels 데이터 배열을 해석할 channel값
	 * @param batches 데이터 배열을 해석할 batch값
	 * @param nane 출력에 사용할 레이블
	 */
	static void printData(const DATATYPE *data, UINT rows, UINT cols, UINT channels,
                          UINT batches, std::string name);
	/**
	 * @details 장치의 메모리를 지정된 데이터 구조로 메타정보와 함께 설정된 출력 스트림에
     *         출력한다.
	 * @param data 출력할 데이터 장치 배열 포인터
	 * @param rows 데이터 배열을 해석할 row값
	 * @param cols 데이터 배열을 해석할 col값
	 * @param channels 데이터 배열을 해석할 channel값
	 * @param batches 데이터 배열을 해석할 batch값
	 * @param nane 출력에 사용할 레이블
	 */
	static void printDeviceData(const DATATYPE *d_data, UINT rows, UINT cols, UINT channels,
            UINT batches, std::string name);
	/**
	 * @details 주어진 메세지를 설정된 출력 스트림에 출력한다.
	 * @param message 설정된 출력 스트림에 출력할 메세지
	 */
	static void printMessage(std::string message);

	static void refineParamName(const char* namePtr, char* tempName);

	/**
	 * @details 설정된 출력 스트림 출력 여부 플래그를 조회한다.
	 * @return 설정된 출력 스트림 출력 여부
	 */
	static int getPrint() { return Util::print; }
	/**
	 * @details 설정된 출력 스트림 출력 여부 플래그를 설정한다.
	 * @param print 설정된 출력 스트림 출력 여부
	 */
	static void setPrint(bool print) { Util::print = print; };
	/**
	 * @details 로그를 출력할 스트림을 설정한다.
	 * @param outstream 로그를 출력할 스트림
	 */
	static void setOutstream(std::ostream *outstream) {
		Util::outstream = outstream;
	}
	/**
	 * @details 로그를 출력할 파일을 설정한다.
	 * @param outfile 로그를 출력할 파일 경로
	 */
	static void setOutstream(std::string outfile) {
		Util::outstream = NULL;
		SNEW(Util::outstream, std::ofstream, outfile.c_str(),
                std::ios::out | std::ios::binary);
		SASSUME0(Util::outstream != NULL);
	}

#ifndef GPU_MODE
	/**
	 * @details 입력의 3차원 행렬을 출력의 3차원 행렬의 구조로 변환하여 출력 행렬에 전달한다.
	 * @param input 입력 3차원 행렬
	 * @param output 출력 3차원 행렬
	 */
	static void convertCube(const rcube &input, rcube &output);
	/**
	 * @details 입력의 데이터에 dropout을 적용한다.
	 * @param input dropout을 적용할 3차원 행렬
	 * @param p_dropout dropout 확률
	 */
	static void dropoutLayer(rcube &input, double p_dropout);
#endif

	/**
	 * @details cudaMalloc Wrapper, 메모리를 할당할 때 마다 잔여 메모리를 로깅한다.
	 * @param devPtr 메모리를 할당할 포인터
	 * @param size 메모리를 할당할 크기
	 */
	template<class T>
	static cudaError_t ucudaMalloc(T **devPtr, size_t size) {
        cudaError_t cudaError = cudaMalloc(devPtr, size);
        cuda_mem += size;
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        return cudaError;
	}

	static void printVramInfo() {
		size_t free, total;
		cudaMemGetInfo(&free, &total);
        std::cout << ":::VRAM INFO:::" << std::endl;
        std::cout << "FREE: << " << free/(1024*1024) << "MB of TOTAL "
            << total/(1024*1024) << "MB" << std::endl;
	}

	static bool temp_flag;					///< 디버깅용 (임시)
	static std::string imagePath;

private:
	static bool print;						///< 로그 출력 여부 플래그
	static size_t cuda_mem;
	static int alloc_cnt;
	static std::ostream *outstream;				///< 로그 출력 스트림
};

#endif /* UTIL_H_ */
