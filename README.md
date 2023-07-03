# C-Torch: A Tensor Library in C

This project is an attempt to create a simple, lightweight tensor library in C that resembles PyTorch.

## Features

- Basic tensor operations (add, subtract, multiply, divide)
- Automatic differentiation (autograd)

## Building the Project

To build the project, you need to have `gcc` and `make` installed on your system. You can build the project by running `make` in the project's root directory.

### Dependencies

The project has a dependency on the Criterion library for testing. Here's how to install it:

**Ubuntu:**

```bash
sudo add-apt-repository ppa:snaipewastaken/ppa
sudo apt-get update
sudo apt-get install criterion-dev
```

**macOS (Homebrew):**

```bash
brew install snaipe/soft/criterion
```

Once Criterion is installed, you can build the test suite by running `make test` in the project's root directory.

## Usage

Here's a basic example of how to use the library:

```c
#include "tensor.h"

int main() {
    int shape[] = {2, 2};
    Tensor* t1 = tensor_create(shape, 2);
    Tensor* t2 = tensor_create(shape, 2);

    tensor_set(t1, (int[]){0, 0}, 1.0);
    tensor_set(t1, (int[]){0, 1}, 2.0);
    tensor_set(t1, (int[]){1, 0}, 3.0);
    tensor_set(t1, (int[]){1, 1}, 4.0);

    tensor_set(t2, (int[]){0, 0}, 5.0);
    tensor_set(t2, (int[]){0, 1}, 6.0);
    tensor_set(t2, (int[]){1, 0}, 7.0);
    tensor_set(t2, (int[]){1, 1}, 8.0);

    Tensor* t3 = tensor_add(t1, t2);

    printf("%f\\n", tensor_get(t3, (int[]){0, 0}));  // Outputs 6.0

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);

    return 0;
}
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you would like to contribute to the project.
