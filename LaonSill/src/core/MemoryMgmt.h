/**
 * @file MemoryMgmt.h
 * @date 2017-12-05
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef MEMORYMGMT_H
#define MEMORYMGMT_H 

#include <stdlib.h>
#include <limits.h>

#include <map>
#include <mutex>


typedef struct MemoryEntry_s {
    char            filename[PATH_MAX];
    char            funcname[PATH_MAX];
    int             line;
    unsigned long   size;   
    uint64_t        index;
    bool            once;
    bool            cpu;
} MemoryEntry;

#define SMALLOC_(obj, type, once, size, filename, func, line)                           \
    do {                                                                                \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = (type*) ptr;                                                          \
            MemoryMgmt::insertEntry(filename, func, line, size, once, ptr, true);       \
        }                                                                               \
    } while (0)

#define SMALLOC_NOLOG_(obj, type, size)                                                 \
    do {                                                                                \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = (type*) ptr;                                                          \
        }                                                                               \
    } while (0)

#define SFREE_(obj)                                                                     \
    do {                                                                                \
        MemoryMgmt::removeEntry((void*)obj);                                            \
        free(obj);                                                                      \
    } while (0)

#define SFREE_NOLOG_(obj)                                                               \
    do {                                                                                \
        free(obj);                                                                      \
    } while (0)

#define SNEW_(obj, type, once, filename, func, line, ...)                               \
    do {                                                                                \
        int size = sizeof( type );                                                      \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = new ((type*)ptr) type(__VA_ARGS__);                                   \
            MemoryMgmt::insertEntry(filename, func, line, size, once, ptr, true);       \
        }                                                                               \
    } while (0)

#define SNEW_NOLOG_(obj, type, ...)                                                     \
    do {                                                                                \
        int size = sizeof( type );                                                      \
        void* ptr = malloc(size);                                                       \
        if (ptr == NULL) {                                                              \
            obj = NULL;                                                                 \
        } else {                                                                        \
            obj = new ((type*)ptr) type(__VA_ARGS__);                                   \
        }                                                                               \
    } while (0)

#define CUDAMALLOC__(devPtr, cudaError, size, filename, func, line)                     \
    do {                                                                                \
        cudaError = cudaMalloc(devPtr, (size_t)size);                                   \
        if (cudaError == cudaSuccess) {                                                 \
            MemoryMgmt::insertEntry(filename, func, line, size, false,                  \
                (void*)(*devPtr), false);                                               \
        }                                                                               \
    } while (0)

#define CUDAMALLOC__NOLOG_(devPtr, cudaError, size)                                     \
    do {                                                                                \
        cudaError = cudaMalloc(devPtr, (size_t)size);                                   \
    } while (0)

#define SDELETE_(obj)                                                                   \
    do {                                                                                \
        MemoryMgmt::removeEntry((void*)obj);                                            \
        delete obj;                                                                     \
    } while (0)

#define SDELETE_NOLOG_(obj)                                                             \
    do {                                                                                \
        delete obj;                                                                     \
    } while (0)

#define CUDAFREE__(dev, ret)                                                            \
    do {                                                                                \
        ret = cudaFree(dev);                                                            \
        if (ret == cudaSuccess) {                                                       \
            MemoryMgmt::removeEntry((void*)dev);                                        \
        }                                                                               \
    } while (0)

#define CUDAFREE__NOLOG_(dev, ret)                                                      \
    do {                                                                                \
        ret = cudaFree(dev);                                                            \
    } while (0)

/* NOTE: we can use __func__ macro because we use C++11 */
#ifdef DEBUG_MODE
#define SMALLOC(obj, type, size)        SMALLOC_(obj, type, false, size, __FILE__, __func__, __LINE__) 
#define SNEW(obj, type, args...)        SNEW_(obj, type, false, __FILE__, __func__, __LINE__, ##args )
#define SMALLOC_ONCE(obj, type, size)   SMALLOC_(obj, type, true, size, __FILE__, __func__, __LINE__) 
#define SNEW_ONCE(obj, type, args...)   SNEW_(obj, type, true, __FILE__, __func__, __LINE__, ##args )
#define CUDAMALLOC_(devPtr, ret, size)  CUDAMALLOC__(devPtr, ret, size, __FILE__, __func__, __LINE__)
#define SDELETE(obj)                    SDELETE_(obj)
#define SFREE(obj)                      SFREE_(obj)
#define CUDAFREE_(dev, ret)             CUDAFREE__(dev, ret)
#else
#define SMALLOC(obj, type, size)        SMALLOC_NOLOG_(obj, type, size)
#define SNEW(obj, type, args...)        SNEW_NOLOG_(obj, type, ##args )
#define SMALLOC_ONCE(obj, type, size)   SMALLOC_NOLOG_(obj, type, size)
#define SNEW_ONCE(obj, type, args...)   SNEW_NOLOG_(obj, type, ##args )
#define CUDAMALLOC_(devPtr, ret, size)  CUDAMALLOC__NOLOG_(devPtr, ret, size)
#define SDELETE(obj)                    SDELETE_NOLOG_(obj)
#define SFREE(obj)                      SFREE_NOLOG_(obj)
#define CUDAFREE_(dev, ret)             CUDAFREE__NOLOG_(dev, ret)
#endif

#define CUDAMALLOC(devPtr, size)                                                        \
    do {                                                                                \
        cudaError_t _cuda_error_ret_;                                                   \
        CUDAMALLOC_(devPtr, _cuda_error_ret_, size);                                    \
        checkCudaErrors(_cuda_error_ret_);                                              \
    } while (0)

#define CUDAFREE(dev)                                                                   \
    do {                                                                                \
        cudaError_t _cuda_error_ret_;                                                   \
        CUDAFREE_(dev, _cuda_error_ret_);                                               \
        checkCudaErrors(_cuda_error_ret_);                                              \
    } while (0)

typedef enum MemoryMgmtSortOption_e {
    MemoryMgmtSortOptionNone = 0,
    MemoryMgmtSortOptionIndex,
    MemoryMgmtSortOptionSize
} MemoryMgmtSortOption;

class MemoryMgmt {
public: 
    MemoryMgmt() {}
    virtual ~MemoryMgmt() {}

    static void init();
    static void insertEntry(const char* filename, const char* funcname, int line,
        unsigned long size, bool once, void* ptr, bool cpu);
    static void removeEntry(void* ptr);
    static void dump(MemoryMgmtSortOption option, bool skipOnce);

private:
    static std::map<void*, MemoryEntry>     entryMap;
    static std::mutex                       entryMutex;
    static uint64_t                         usedMemTotalSize;
    static uint64_t                         currIndex;
    static uint64_t                         usedGPUMemTotalSize;
};

#endif /* MEMORYMGMT_H */
