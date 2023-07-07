#ifndef TENSOR_H
#define TENSOR_H
#include <stdbool.h>


// Global error state

extern int tensor_errno;
const char* SHAPE_ERR_MSG;
#define TENSOR_SUCESS 0
#define TENSOR_INVALID_SHAPE 1

typedef struct Tensor {
    int* shape;          // Array representing the shape of the tensor
    int rank;            // Number of dimensions
    float* data;         // Array to hold tensor data
    bool requires_grad;  // Boolean flag to indicate if this tensor needs gradients
    float* grad;         // Array to hold tensor gradient
    struct Tensor* inputs[2];  // Input tensors
    void (*grad_fn)(struct Tensor*, struct Tensor*); // Function pointer for the gradient function
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
bool validate_same_shape(Tensor* tensor1, Tensor* tensor2);


#endif // TENSOR_H