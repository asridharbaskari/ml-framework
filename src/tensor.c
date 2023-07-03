#include "tensor.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>


// Global error state
int tensor_errno = 0;
const char* SHAPE_ERR_MSG = "Tensors have incompatible shape\n";
#define TENSOR_SUCESS 0
#define TENSOR_INVALID_SHAPE 1

// Helper function to calculate the total size of the tensor
int tensor_size(Tensor* tensor) {
    int size = 1;
    for (int i = 0; i < tensor->rank; i++) {
        size *= tensor->shape[i];
    }
    return size;
}
// Helper function to validate if two tensors have the same shape
bool validate_same_shape(Tensor* tensor1, Tensor* tensor2) {
    if (tensor1->rank != tensor2->rank) {
        return false;
    }
    for (int i = 0; i < tensor1->rank; i++) {
        if (tensor1->shape[i] != tensor2->shape[i]) {
            return false;
        }
    }
    return true;
}
// Helper function to compute the flattened index from a list of indices
int calculate_index(Tensor* tensor, int* indices) {
    int index = 0;
    int size = 1;
    for (int i = tensor->rank - 1; i >= 0; i--) {
        index += size * indices[i];
        size *= tensor->shape[i];
    }
    return index;
}

void index_to_indices(int index, int* indices, int* shape, int rank) {
    for (int i = rank - 1; i >= 0; i--) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
}

Tensor* tensor_create(int* shape, int rank, bool requires_grad) {
    // Allocate memory for the tensor structure
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));

    // Set the rank
    tensor->rank = rank;

    // Allocate memory and set the shape
    tensor->shape = (int*)malloc(rank * sizeof(int));
    for (int i = 0; i < rank; i++) {
        tensor->shape[i] = shape[i];
    }

    // Calculate the total size
    int size = tensor_size(tensor);

    // Allocate memory and initialize the data to zeros
    tensor->data = (float*)calloc(size, sizeof(float));
    tensor->grad = NULL;
    tensor->requires_grad = requires_grad;

    return tensor;
}

void tensor_free(Tensor* tensor) {
    // Free the memory allocated for the shape and data
    free(tensor->shape);
    free(tensor->data);

    // Free the memory allocated for the tensor itself
    free(tensor);
}

float tensor_get(Tensor* tensor, int* indices) {
    int index = calculate_index(tensor, indices);
    return tensor->data[index];
}

void tensor_set(Tensor* tensor, int* indices, float value) {
    // Compute the index using flattening rules
    int index = calculate_index(tensor, indices);
    tensor->data[index] = value;
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
    for (int i = 0; i < size; i++) {
        result->data[i] = tensor1->data[i] / tensor2->data[i];
    }

    return result;
}
