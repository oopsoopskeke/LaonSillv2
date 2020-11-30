/**
 * @file Perf.h
 * @date 2016-11-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PERF_H
#define PERF_H 

#include <string>
#include <map>

#include "common.h"
#include "PerfList.h"

#define SPERF_MARK(name, args...)               PerfList::mark##name(##args)      
#define SPERF_START(name, sTimePtr)             PerfList::start##name(sTimePtr)
#define SPERF_END(name, sTimeVal, args...)      PerfList::end##name(sTimeVal, ##args)
#define SPERF_COUNT(name)                       PerfList::_##name##Count
#define SPERF_TIME(name)                        PerfList::_##name##Time
#define SPERF_AVGTIME(name)                     PerfList::_##name##AvgTime
#define SPERF_MAXTIME(name)                     PerfList::_##name##MaxTime
#define SPERF_VALUE(name, aname)                PerfList::_##name##_##aname
#define SPERF_CLEAR(name)                       PerfList::clear##name()

class Perf {
public: 

                    Perf() {}
    virtual        ~Perf() {}
    static void     init();

    static bool     isPerfExist(std::string perfName);

    static char*    getDesc(std::string perfName);
    static bool     isJobScope(std::string perfName);
    static bool     isUseTime(std::string perfName);
    static bool     isUseAvgTime(std::string perfName);
    static bool     isUseMaxTime(std::string perfName);
    static long     getCount(std::string perfName);
    static double   getTime(std::string perfName);
    static double   getAvgTime(std::string perfName);
    static double   getMaxTime(std::string perfName);
    static int      getArgCount(std::string perfName);
    static void     getArgValue(std::string perfName, int argIndex, void* value);
    static char*    getArgDesc(std::string perfName, int argIndex);
    static char*    getArgName(std::string perfName, int argIndex);
    static char*    getArgTypeName(std::string perfName, int argIndex);
    static PerfArgDef::PerfArgType
                    getArgType(std::string perfName, int argIndex);

private:
    static std::map<std::string, PerfDef*>      perfDefMap;
};

#endif /* PERF_H */
