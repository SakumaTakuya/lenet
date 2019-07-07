
# Makefile

CFLAGS = -I. -O0

lenet: main.cu cnnfunc.cu 
	nvcc $(CFLAGS) -o lenet main.cu cnnfunc.cu

clean:
	rm -f lenet

.PHONY: clean

