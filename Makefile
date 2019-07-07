
# Makefile

CFLAGS = -I. -O0

lenet: main.cu gpu_layer.cu cnnfunc.cu 
	nvcc $(CFLAGS) -o lenet main.cu gpu_layer.cu cnnfunc.cu

clean:
	rm -f lenet

.PHONY: clean

