#ifndef RAGEX_TYPES_H
#define RAGEX_TYPES_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    DTYPE_F32,
    DTYPE_INT8,
    DTYPE_INT4
} DataType;

typedef struct {
    int ndim;
    int shape[4];
    DataType dtype;
    void* data;
} Tensor;

#endif // RAGEX_TYPES_H
