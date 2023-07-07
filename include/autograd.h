#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

typedef void (*grad_fn)(Tensor* result, Tensor* grad_output);

int tensor_size(Tensor* tensor);
bool validate_same_shape(Tensor* tensor1, Tensor* tensor2);
int calculate_index(Tensor* tensor, int* indices);
void index_to_indices(int index, int* indices, int* shape, int rank);
Tensor* tensor_create(int* shape, int rank, bool requires_grad);
void tensor_free(Tensor* tensor);
float tensor_get(Tensor* tensor, int* indices);
void tensor_set(Tensor* tensor, int* indices, float value);
Tensor* tensor_add(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_subtract(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_multiply(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_divide(Tensor* tensor1, Tensor* tensor2);
void zero_grad(Tensor* tensor);
void tensor_backward(Tensor* tensor, Tensor* grad_output);
void setup_grad_fn(Tensor* result, Tensor* input1, Tensor* input2, grad_fn grad_func);

#endif // AUTOGRAD_H
