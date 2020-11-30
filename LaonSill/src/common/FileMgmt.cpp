/**
 * @file FileMgmt.cpp
 * @date 2016-10-31
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>

#include "FileMgmt.h"
#include "SysLog.h"

using namespace std;

// in glibc-2.7, the longest error message string is 50 character..
// therefore 256 character size is enough,...
// otherwise, please fix me :)
#define ERROR_BUFFER_SIZE   (256)

void FileMgmt::checkDir(const char* path) {

    // XXX: 일단 에러처리를 assert로만 했음.
    struct  stat st;
    if (stat(path, &st) == -1) {
        int err = errno;

        if (err == ENOENT) {
            FileMgmt::makeDir(path);
        } else {
            char    errBuf[ERROR_BUFFER_SIZE];
            // XXX: we only consider XSI-compliant version
            // XXX: we do not consider strerror_r() makes an error again...
            SASSERT0(strerror_r(err, errBuf, ERROR_BUFFER_SIZE) == 0);
            SYS_LOG("checkDir() is failed. path=%s,errno=%d,reason=%s", path, err, errBuf);
            SASSERT0(0);
        }
    }
}

int FileMgmt::openFile(const char* path, int flag) {
    int fd;
    
    while (true) {
        fd = open(path, flag, 0664);
        int err = errno;

        if ((fd == -1) && ((err == EINTR) || (err == EWOULDBLOCK))) // try again
            continue;
        else
            break;
    }
        
    if (fd == -1) {
        int     err = errno;
        char    errBuf[ERROR_BUFFER_SIZE];
        SASSERT0(strerror_r(err, errBuf, ERROR_BUFFER_SIZE) == 0);
        SYS_LOG("openFile() is failed. path=%s,flag=%d,errno=%d,reason=%s",
            path, flag, err, errBuf);
        SASSERT0(0);
    }

    return fd;
}

void FileMgmt::removeFile(const char* path) {
    int ret = remove(path);
    if (ret == -1) {
        int err = errno;
        SYS_LOG("removeFile() is failed. path=%s,errno=%d", path, errno);
        SASSERT0(0);
    }
}

void FileMgmt::makeDir(const char* path) { 
    if (mkdir(path, 0700) == -1) {
        int     err = errno;
        char    errBuf[ERROR_BUFFER_SIZE];
        SASSERT0(strerror_r(err, errBuf, ERROR_BUFFER_SIZE) == 0);
        SYS_LOG("makeDir() is failed. path=%s,errno=%d,reason=%s", path, err, errBuf);

        SASSERT0(0);
    }
}
