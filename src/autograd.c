#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

int tensor_errno = 0;
const char* SHAPE_ERR_MSG = "Tensors have incompatible shape\n";
// Gradient function for binary operations
typedef void (*grad_fn)(Tensor* result, Tensor* grad_output);

// Utility function to zero out the gradient of a tensor
void zero_grad(Tensor* tensor) {
    if (tensor->grad != NULL) {
        free(tensor->grad);
        tensor->grad = NULL;
    }
}

void tensor_backward(Tensor* tensor, Tensor* grad_output) {
    if (!tensor->requires_grad) {
        return;
    }

    if (tensor->grad == NULL) {
        tensor->grad = tensor_create(tensor->shape, tensor->rank, false);
    }

    int size = tensor_size(tensor);
    for (int i = 0; i < size; i++) {
        (tensor->grad[i]) += grad_output->data[i];
    }

    if (tensor->grad_fn != NULL) {
        tensor->grad_fn(tensor, grad_output);
    }
}

// Implementing backward functions for our operations
void add_backward(Tensor* result, Tensor* grad_output) {
    tensor_backward(result->inputs[0], grad_output);
    tensor_backward(result->inputs[1], grad_output);
}

void subtract_backward(Tensor* result, Tensor* grad_output) {
    tensor_backward(result->inputs[0], grad_output);

    // Create a tensor for negative grad_output
    Tensor* neg_grad_output = tensor_create(grad_output->shape, grad_output->rank, false);
    int size = tensor_size(grad_output);
    for (int i = 0; i < size; i++) {
        neg_grad_output->data[i] = -grad_output->data[i];
    }

    tensor_backward(result->inputs[1], neg_grad_output);

    // Free the created tensor
    tensor_free(neg_grad_output);
}

void multiply_backward(Tensor* result, Tensor* grad_output) {
    Tensor* input1_grad = tensor_multiply(result->inputs[1], grad_output);
    Tensor* input2_grad = tensor_multiply(result->inputs[0], grad_output);

    tensor_backward(result->inputs[0], input1_grad);
    tensor_backward(result->inputs[1], input2_grad);

    tensor_free(input1_grad);
    tensor_free(input2_grad);
}

void divide_backward(Tensor* result, Tensor* grad_output) {
    Tensor* input1_grad = tensor_divide(grad_output, result->inputs[1]);

    Tensor* denominator_squared = tensor_multiply(result->inputs[1], result->inputs[1]);
    Tensor* temp = tensor_divide(result->inputs[0], denominator_squared);
    Tensor* input2_grad = tensor_multiply(temp, grad_output);

    tensor_backward(result->inputs[0], input1_grad);
    tensor_backward(result->inputs[1], input2_grad);

    tensor_free(input1_grad);
    tensor_free(input2_grad);
    tensor_free(denominator_squared);
    tensor_free(temp);
}

void setup_grad_fn(Tensor* result, Tensor* input1, Tensor* input2, grad_fn grad_func) {
    result->inputs[0] = input1;
    result->inputs[1] = input2;
    result->grad_fn = grad_func;
}
Tensor* tensor_add(Tensor* tensor1, Tensor* tensor2) {
    if (!validate_same_shape(tensor1, tensor2)) {
        printf("%s", SHAPE_ERR_MSG);
        tensor_errno = TENSOR_INVALID_SHAPE;
        return NULL;
    }
    bool requires_grad = tensor1->requires_grad || tensor2->requires_grad;
    Tensor* result = tensor_create(tensor1->shape, tensor1->rank, requires_grad);
    int size = tensor_size(tensor1);
    if (requires_grad) {
        setup_grad_fn(result, tensor1, tensor2, add_backward);
    }
    return result;
    for (int i = 0; i < size; i++) {
        result->data[i] = tensor1->data[i] + tensor2->data[i];
    }

    return result;
}

Tensor* tensor_subtract(Tensor* tensor1, Tensor* tensor2) {
    if (!validate_same_shape(tensor1, tensor2)) {
        printf("%s", SHAPE_ERR_MSG);
        tensor_errno = TENSOR_INVALID_SHAPE;
        return NULL;
    }
    bool requires_grad = tensor1->requires_grad || tensor2->requires_grad;
    Tensor* result = tensor_create(tensor1->shape, tensor1->rank, requires_grad);

    int size = tensor_size(tensor1);
     if (requires_grad) {
        setup_grad_fn(result, tensor1, tensor2, subtract_backward);
    }
    for (int i = 0; i < size; i++) {
        result->data[i] = tensor1->data[i] - tensor2->data[i];
    }
    return result;
}

Tensor* tensor_multiply(Tensor* tensor1, Tensor* tensor2) {
    if (!validate_same_shape(tensor1, tensor2)) {
        printf("%s", SHAPE_ERR_MSG);
        tensor_errno = TENSOR_INVALID_SHAPE;
        return NULL;
    }
    bool requires_grad = tensor1->requires_grad || tensor2->requires_grad;
    Tensor* result = tensor_create(tensor1->shape, tensor1->rank, requires_grad);
    int size = tensor_size(tensor1);
       if (requires_grad) {
        setup_grad_fn(result, tensor1, tensor2, multiply_backward);
    }

    for (int i = 0; i < size; i++) {
        result->data[i] = tensor1->data[i] * tensor2->data[i];
    }

    return result;
}

Tensor* tensor_divide(Tensor* tensor1, Tensor* tensor2) {
    if (!validate_same_shape(tensor1, tensor2)) {
        printf("%s", SHAPE_ERR_MSG);
        tensor_errno = TENSOR_INVALID_SHAPE;
        return NULL;
    }
    bool requires_grad = tensor1->requires_grad || tensor2->requires_grad;
    Tensor* result = tensor_create(tensor1->shape, tensor1->rank, requires_grad);
    int size = tensor_size(tensor1);
       if (requires_grad) {
        setup_grad_fn(result, tensor1, tensor2, divide_backward);
    }
    for (int i = 0; i < size; i++) {
        result->data[i] = tensor1->data[i] / tensor2->data[i];
    }

    return result;
}
