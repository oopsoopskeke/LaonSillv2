/**
 * @file common.cpp
 * @date 2017-07-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"

using namespace std;

const char* LAONSILL_BUILD_PATH_ENVNAME = "LAONSILL_BUILD_PATH";
extern const char* LAONSILL_HOME_ENVNAME;

string Common::GetSoooARelPath(string path) {
    if (getenv(LAONSILL_BUILD_PATH_ENVNAME) != NULL) {
        return string(getenv(LAONSILL_BUILD_PATH_ENVNAME)) + "/src/" + path;
    } else if (getenv(LAONSILL_HOME_ENVNAME) != NULL) {
        return string(getenv(LAONSILL_HOME_ENVNAME)) + "/" + path;
    } else {
        return path;
    }
}
