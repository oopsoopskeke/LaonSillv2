/**
 * @file InitParam.h
 * @date 2016-10-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef INITPARAM_H
#define INITPARAM_H 

#include "common.h"

class InitParam {
public:
                        InitParam() {}
    virtual            ~InitParam() {}

    static void         init();
    static void         destroy();

private:
    static bool         isWhiteSpace(char c);
    static void         setParam(const char* paramName, const char* paramValue);
    static void         loadParam(const char* line, int len);
    static void         load();
};

#endif /* INITPARAM_H */
