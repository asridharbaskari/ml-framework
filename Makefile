CC = gcc
CFLAGS = -Iinclude -I/opt/homebrew/Cellar/criterion/2.4.1_2/include -Wall -Wextra -O2
LDFLAGS = -L/opt/homebrew/Cellar/criterion/2.4.1_2/lib
CRITERION_FLAGS = -lcriterion

# Gather all .c files in the src/ directory
SOURCES = $(wildcard src/*.c)
# Replace the .c extension of source files with .o for object files
OBJECTS = $(SOURCES:.c=.o)
# Gather all test .c files in the test/ directory
TESTS = $(wildcard test/test_*.c)
# Replace the .c extension of test files with .out for test executables
TEST_EXECUTABLES = $(TESTS:.c=.out)

.PHONY: all clean tests

all: tests

tests: $(TEST_EXECUTABLES)
	for exec in $(TEST_EXECUTABLES); do \
	    ./$$exec; \
	done

test/%.out: test/%.c $(OBJECTS)
	$(CC) $< $(OBJECTS) $(CFLAGS) $(LDFLAGS) $(CRITERION_FLAGS) -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f src/*.o test/*.out
