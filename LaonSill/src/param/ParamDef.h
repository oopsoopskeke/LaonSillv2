/**
 * @file ParamDef.h
 * @date 2016-10-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PARAMDEF_H
#define PARAMDEF_H 

#include <string.h>

#include "common.h"

#define PARAMDEF_DESC_MAXSIZE           (256)
#define PARAMDEF_DEFAULTVALUE_MAXSIZE   (128)
#define PARAMDEF_TYPENAME_MAXSIZE       (32)

class ParamDef {
public:
    ParamDef(const char* desc, const char* defaultValue, const char* typeName,
        bool isMandatory, bool isMutable, bool isSessScope, void* valuePt, int valueLen) {
        strcpy(this->desc, desc);
        strcpy(this->defaultValue, defaultValue);
        strcpy(this->typeName, typeName);
        this->isMandatory = isMandatory;
        this->isMutable = isMutable;
        this->isSessScope = isSessScope;
        this->valuePt = valuePt;
        this->valueLen = valueLen;
    }
    virtual    ~ParamDef() {}

    // use "string" instead of "fixed char array"?
    char        desc[PARAMDEF_DESC_MAXSIZE];
    char        defaultValue[PARAMDEF_DEFAULTVALUE_MAXSIZE];
    char        typeName[PARAMDEF_TYPENAME_MAXSIZE];
    bool        isMandatory;
    bool        isMutable;
    bool        isSessScope;

    void*       valuePt;
    int         valueLen;
};

#endif /* PARAMDEF_H */
