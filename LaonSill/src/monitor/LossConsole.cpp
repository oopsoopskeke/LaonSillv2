/**
 * @file LossConsole.cpp
 * @date 2017-06-17
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "LossConsole.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "Perf.h"

using namespace std;

LossConsole::LossConsole(vector<string> lossNames) {
    for (int i = 0; i < lossNames.size(); i++) {
        LossMovingAvg elem;
        elem.count = 0;
        elem.avg = 0.0;
        elem.lossName = lossNames[i];
        lossMovingAvgs.push_back(elem);
    }
}

void LossConsole::addValue(int index, float value) {
    SASSUME0(index < lossMovingAvgs.size());

    LossMovingAvg *lma = &lossMovingAvgs[index];
    lma->avg += (value - lma->avg) / (float)(lma->count + 1);
    lma->count += 1;
}

void LossConsole::printLoss(FILE* fp) {
    for (int i = 0; i < lossMovingAvgs.size(); i++) {
        if (fp == stdout) {
            STDOUT_LOG("average loss[%s] : %f", lossMovingAvgs[i].lossName.c_str(),
                lossMovingAvgs[i].avg);
        } else {
            fprintf(fp, "average loss[%s] : %f\n", lossMovingAvgs[i].lossName.c_str(),
                lossMovingAvgs[i].avg);
        }

    }

    if (SPARAM(INPUT_DATA_PROVIDER_MEASURE_PERFORMANCE)) {
        STDOUT_BLOCK(cout << "total time : " << SPERF_TIME(DATAINPUT_ACCESS_TIME)
            << ", avg time : " << SPERF_AVGTIME(DATAINPUT_ACCESS_TIME)
            << ", max time : " << SPERF_MAXTIME(DATAINPUT_ACCESS_TIME)
            << endl;);
        SPERF_CLEAR(DATAINPUT_ACCESS_TIME);
    }
}

void LossConsole::clear() {
    for (int i = 0; i < lossMovingAvgs.size(); i++) {
        lossMovingAvgs[i].count = 0;
        lossMovingAvgs[i].avg = 0.0;
    }
}
