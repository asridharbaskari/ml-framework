#include <criterion/criterion.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"

#define LARGE_DIMENSION 1000
#define NUM_RANDOM_TESTS 100
#define MAX_RANK 5
#define MAX_DIMENSION_SIZE 10

// This helper function generates a random tensor
Tensor* generate_random_tensor() {
    int rank = MAX_RANK;
    int* shape = malloc(rank * sizeof(int));
    for (int i = 0; i < rank; i++) {
        shape[i] = MAX_DIMENSION_SIZE;
    }

    Tensor* tensor = tensor_create(shape, rank, true);
    int size = tensor_size(tensor);
    for (int i = 0; i < size; i++) {
        int* indices = malloc(rank * sizeof(int));
        index_to_indices(i, indices, shape, rank);
        tensor_set(tensor, indices, (float)rand() / RAND_MAX);
        free(indices);
    }

    free(shape);
    return tensor;
}

void setup() {
    srand(time(NULL));
}

void teardown() {
    // Nothing to teardown in this case
}

Test(tensor, random_tests, .init = setup, .fini = teardown) {
    for (int i = 0; i < NUM_RANDOM_TESTS; i++) {
        Tensor* tensor1 = generate_random_tensor();
        Tensor* tensor2 = generate_random_tensor();
        
        // Add
        Tensor* result_add = tensor_add(tensor1, tensor2);
        int size = tensor_size(tensor1);
        for (int i = 0; i < size; i++){
            int* indices = malloc(tensor1->rank * sizeof(int));
            index_to_indices(i, indices, tensor1->shape, tensor1->rank);
            cr_assert_float_eq(tensor_get(result_add, indices), tensor_get(tensor1, indices) + tensor_get(tensor2, indices), 1e-6);
            free(indices);
        }


        // Subtract
        Tensor* result_sub = tensor_subtract(tensor1, tensor2);
        for(int i = 0; i < size; i++){
            int* indices = malloc(tensor1->rank * sizeof(int));
            index_to_indices(i, indices, tensor1->shape, tensor1->rank);
            cr_assert_float_eq(tensor_get(result_sub, indices), tensor_get(tensor1, indices) - tensor_get(tensor2, indices), 1e-6);
            free(indices);
        }

        // Multiply
        Tensor* result_mul = tensor_multiply(tensor1, tensor2);
        for(int i = 0; i < size; i++){
            int* indices = malloc(tensor1->rank * sizeof(int));
            index_to_indices(i, indices, tensor1->shape, tensor1->rank);
            cr_assert_float_eq(tensor_get(result_mul, indices), tensor_get(tensor1, indices) * tensor_get(tensor2, indices), 1e-6);
            free(indices);
        }

        // Divide
        Tensor* result_div = tensor_divide(tensor1, tensor2);
        for(int i = 0; i < size; i++){
            int* indices = malloc(tensor1->rank * sizeof(int));
            index_to_indices(i, indices, tensor1->shape, tensor1->rank);
            cr_assert_float_eq(tensor_get(result_div, indices), tensor_get(tensor1, indices) / tensor_get(tensor2, indices), 1e-6);
            free(indices);
        }

        tensor_free(tensor1);
        tensor_free(tensor2);
    }
}

Test(tensor, stress_test, .init = setup, .fini = teardown) {
    int shape[] = {LARGE_DIMENSION, LARGE_DIMENSION};
    Tensor* tensor1 = tensor_create(shape, 2, true);
    Tensor* tensor2 = tensor_create(shape, 2, true);
    // Perform various operations and assertions on tensor1 and tensor2
    // Just an example with addition
    Tensor* result_add = tensor_add(tensor1, tensor2);
    for(int i = 0; i < tensor1->rank; i++){
        int* indices = malloc(tensor1->rank * sizeof(int));
        index_to_indices(i, indices, tensor1->shape, tensor1->rank);
        cr_assert_float_eq(tensor_get(result_add, indices), tensor_get(tensor1, indices) + tensor_get(tensor2, indices), 1e-6);
        free(indices);
    }

    tensor_free(tensor1);
    tensor_free(tensor2);
}
