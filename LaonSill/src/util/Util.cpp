/*
 * Util.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include <cstdlib>
#include <stdarg.h>
#include <cstring>

#include "Util.h"
#include "SysLog.h"

using namespace std;

bool Util::temp_flag = false;
bool Util::print = false;
size_t Util::cuda_mem = 0;
int Util::alloc_cnt = 0;
ostream *Util::outstream = &cout;
string Util::imagePath = "";
//int Util::batchCount = 0;

static const char *LEVEL_LABEL[4] = {"DEBUG","INFO ", "WARN ", "ERROR"};

void log_print(FILE *fp, int log_level, const char* filename, const int line,
    const char *func, char *fmt, ...) {
	if(fp != NULL) {
		char log[1024];
		int log_index = 0;
		time_t t;
		va_list list;

		time(&t);

		sprintf(log, "[%s", ctime(&t));
		log_index = strlen(log)-1;

		sprintf(&log[log_index], "] (%s) %s:%d %s(): ", LEVEL_LABEL[log_level], filename,
            line, func);
		log_index = strlen(log);

		va_start(list, fmt);
		vsprintf(&log[log_index], fmt, list );
		va_end(list);

		fprintf(fp, "%s", log);
		fputc('\n', fp);

		fflush(fp);
	}
}

int Util::random(int min, int max)
{
	return (int)rand() * (max-min) + min;
}


int Util::pack4BytesToInt(unsigned char *buffer)
{
	int result = 0;
	for(int i = 0; i < 4; i++) {
		result += buffer[i] << 8*(3-i);
	}
	return result;
}


uint32_t Util::vecCountByAxis(const vector<uint32_t>& vec, const uint32_t axis) {
	const uint32_t vecSize = vec.size();
	assert(axis < vecSize);

	uint32_t count = 1;
	for (uint32_t i = axis; i < vecSize; i++) {
		if (vec[i] > 0)
			count *= vec[i];
	}
	return count;
}

void Util::saveStringToFstream(ofstream& ofs, const string& str) {
	size_t strLength = str.size();
	ofs.write((char*)&strLength, sizeof(size_t));
	ofs.write((char*)str.c_str(), strLength);
}

void Util::loadStringFromFstream(ifstream& ifs, string& str) {
	size_t strLength;

    SASSERT0(ifs.is_open());
	ifs.read((char*)&strLength, sizeof(size_t));

	char* str_c = NULL;
	SMALLOC(str_c, char, (strLength + 1) * sizeof(char));
	SASSUME0(str_c != NULL);

	ifs.read(str_c, strLength);
	str_c[strLength] = '\0';

	str = str_c;

	SFREE(str_c);
}

#ifndef GPU_MODE
void Util::printVec(const rvec &vector, string name) {
	if (Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &vector << endl;
		(*outstream) << "rows x cols: " << vector.n_rows << " x " << vector.n_cols << endl;
		vector.print("vec values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printMat(const rmat &matrix, string name) {
	if (Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &matrix << endl;
		(*outstream) << "rows x cols: " << matrix.n_rows << " x " << matrix.n_cols << endl;
		matrix.raw_print((*outstream), "mat values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printCube(const rcube &c, string name) {
	if (Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &c << endl;
		(*outstream) << "rows x cols x slices: " << c.n_rows << " x " << c.n_cols
            << " x " << c.n_slices << endl;
		c.raw_print((*outstream), "cube values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printUCube(const ucube &c, string name) {
	if (Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &c << endl;
		(*outstream) << "rows x cols x slices: " << c.n_rows << " x " << c.n_cols
            << " x " << c.n_slices << endl;
		c.raw_print((*outstream), "cube values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}
#endif

void Util::printData(const DATATYPE *data, UINT rows, UINT cols, UINT channels,
    UINT batches, string name) {
	if (Util::print && data) {
		UINT i,j,k,l;

		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "rows x cols x channels x batches: " << rows << " x " << cols
            << " x " << channels << " x " << batches << endl;

		UINT batchElem = rows*cols*channels;
		UINT channelElem = rows*cols;
		for (i = 0; i < batches; i++) {
			for (j = 0; j < channels; j++) {
				for (k = 0; k < rows; k++) {
					for (l = 0; l < cols; l++) {
						(*outstream) << data[i*batchElem + j*channelElem + l*rows + k] << ", ";
					}
					(*outstream) << endl;
				}
				(*outstream) << endl;
			}
			(*outstream) << endl;
		}

		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printDeviceData(const DATATYPE *d_data, UINT rows, UINT cols, UINT channels,
    UINT batches, string name) {
	if (Util::print) {
		DATATYPE *data = NULL;
		SNEW(data, DATATYPE, rows * cols * channels * batches * sizeof(DATATYPE));
		SASSUME0(data != NULL);
		checkCudaErrors(cudaMemcpyAsync(data, d_data,
            sizeof(DATATYPE)*rows*cols*channels*batches, cudaMemcpyDeviceToHost));
		Util::printData(data, rows, cols, channels, batches, name);
	}
}

void Util::printMessage(string message) {
	if(true || Util::print) {
		(*outstream) << message << endl;
	}
}


void Util::refineParamName(const char* namePtr, char* tempName) {
	int i = 0;

	// 이름이 ' :'를 만나서 끝난 케이스, postfix있음, postfix 제거 후 사용
	// 이름이 '\0'을 만나서 끝난 케이스, postfix없음, 그대로 사용
	while (true) {
		if (namePtr[i] == ':') {
			// tempName 뒤에 _filter, _bias를 붙이고 종료

			int tempi = i;
			while (namePtr[i] != '\0')
				i++;
			while (namePtr[i] != '_')
				i--;

			while (true) {
				tempName[tempi] = namePtr[i];
				if (namePtr[i] == '\0')
					break;
				i++;
				tempi++;
			}
			break;
		}

		tempName[i] = namePtr[i];

		if (namePtr[i] == '\0') {
			break;
		}
		i++;
	}
}

#ifndef GPU_MODE
void Util::convertCube(const rcube &input, rcube &output) {
	// input, output의 dim이 동일한 경우, 변환이 필요없음, input을 output으로 그대로 전달
	if(size(input) == size(output)) {
		output = input;
		return;
	}

	// 두 매트릭스의 elem의 수가 같아야 함
	// 둘 중 하나는 vector여야 함 (vector로, 또는 vector로부터의 변환만 현재 지원)
	if(input.size() != output.size() ||
			!((input.n_cols==1&&input.n_slices==1)||(output.n_cols==1&&output.n_slices==1))) {
		throw Exception();
	}

	// output이 vector인 경우
	if(output.n_cols==1&&output.n_slices==1) {
		output = reshape(input, output.size(), 1, 1, 1);
		return;
	}

	// input이 vector인 경우
	if(input.n_cols==1&&input.n_slices==1) {
		rcube temp = reshape(input, output.n_cols, output.n_rows, output.n_slices);
		for(unsigned int i = 0; i < output.n_slices; i++) {
			output.slice(i) = temp.slice(i).t();
		}
		return;
	}
}

void Util::dropoutLayer(rcube &input, double p_dropout) {
	rcube p = randu<rcube>(size(input));

	UINT slice, row, col;
	for (slice = 0; slice < input.n_slices; slice++) {
		for (row = 0; row < input.n_rows; row++) {
			for (col = 0; col < input.n_cols; col++) {
				if (C_MEM(p, row, col, slice) < p_dropout)
                    C_MEMPTR(input, row, col, slice) = 0;
			}
		}
	}
}

#endif
