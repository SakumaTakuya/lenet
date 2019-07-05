
# Makefile

CFLAGS = -I. -O0

lenet: main.cu gpu_layer.cu cnn_func.cu 
	nvcc $(CFLAGS) -o lenet main.cu gpu_lyer.cu cnnfunc.cu

clean:
	rm -f lenet

.PHONY: clean

