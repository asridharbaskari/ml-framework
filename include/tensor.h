#ifndef TENSOR_H
#define TENSOR_H
#include <stdbool.h>

// Tensor data structure
typedef struct Tensor {
    float* data;
    int* shape;
    int rank;
    struct Tensor* grad;        // Gradient of the tensor
    bool requires_grad;         // Does this tensor require gradients
    struct Operation* creator;  // Operation that created this tensor
} Tensor;

typedef struct Operation {
    Tensor** inputs;
    Tensor* outputs;
    void (*grad_fn) (struct Operation*, struct Tensor*);
} Operation;

Tensor* tensor_create(int* shape, int rank, bool requires_grad);
void tensor_free(Tensor* tensor);
float tensor_get(Tensor* tensor, int* indices);
void tensor_set(Tensor* tensor, int* indices, float value);
Tensor* tensor_add(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_subtract(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_multiply(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_divide(Tensor* tensor1, Tensor* tensor2);

// Helper functions
void index_to_indices(int index, int* indices, int* shape, int rank);
int tensor_size(Tensor* tensor);


#endif // TENSOR_H