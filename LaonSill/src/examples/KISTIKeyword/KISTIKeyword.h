/**
 * @file KISTIKeyword.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef KISTIKEYWORD_H
#define KISTIKEYWORD_H 

#include <vector>

#include "KistiInputLayer.h"

typedef struct top10Sort_s {
    float value;
    int index;

    bool operator < (const struct top10Sort_s &x) const {
        return value < x.value;
    }
} top10Sort;

template<typename Dtype>
class KISTIKeyword {
public: 
    KISTIKeyword() {}
    virtual ~KISTIKeyword() {}

    static void run();
private:
    static int getTop10GuessSuccessCount(const float* data, const float* label, int batchCount,
        int depth, bool train, int epoch, const float* image, int imageBaseIndex,
        std::vector<KistiData> etriData);
    static float getTopKAvgPrecision(int topK, const float* data, const float* label,
        int batchCount, int depth);
};

#endif /* KISTIKEYWORD_H */
