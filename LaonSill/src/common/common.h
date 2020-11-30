/*
 * common.h
 *
 *  Created on: 2016. 8. 27.
 *      Author: jhkim
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>

#include <iostream>
#include <string>

#define ALIGNUP(x, n)               ((~(n-1))&((x)+(n-1)))
#define ALIGNDOWN(x, n)             ((~(n-1))&(x))

#define Nop() 						do {} while(0)

#define SPATH(path)                 (Common::GetSoooARelPath(path))


#define TEST_MODE					0

class Common {
public:
    static std::string GetSoooARelPath(std::string path);

    Common() {}
    virtual ~Common() {}
};

#endif /* COMMON_H_ */
