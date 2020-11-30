/**
 * @file LossConsole.h
 * @date 2017-06-17
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LOSSCONSOLE_H
#define LOSSCONSOLE_H 

#include <vector>
#include <string>

typedef struct LossMovingAvg_s {
    std::string lossName;
    int         count;
    float       avg;
} LossMovingAvg;

class LossConsole {
public: 
    LossConsole(std::vector<std::string> lossNames);
    virtual ~LossConsole() {}

    void addValue(int index, float value);
    void printLoss(FILE* fp);
    void clear();

private:
    std::vector<LossMovingAvg>  lossMovingAvgs;
};

#endif /* LOSSCONSOLE_H */
