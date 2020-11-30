/**
 * @file MeasureManager.h
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief Measure Entry를 관리한다.
 * @details
 */

#ifndef MEASUREMANAGER_H
#define MEASUREMANAGER_H 

#include <vector>
#include <mutex>
#include <string>
#include <map>

#include "MeasureEntry.h"


class MeasureManager {
public: 
    MeasureManager() {}
    virtual ~MeasureManager() {}

    static void init();
    static void insertEntryEx(std::string networkID, std::vector<std::string> itemNames,
        MeasureOption option, int queueSize);
    static void insertEntry(std::string networkID, std::vector<std::string> itemNames);
    static void removeEntry(std::string networkID);
    static MeasureEntry* getMeasureEntry(std::string networkID);
    static void releaseMeasureEntry(std::string networkID);

private:
    static std::map<std::string, MeasureEntry*> entryMap;
    static std::mutex                           entryMapMutex;

};

#endif /* MEASUREMANAGER_H */
