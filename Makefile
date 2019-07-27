
# Makefile

CFLAGS = -I. -O0

lenet: main.cu cnnfunc.cu 
	nvcc $(CFLAGS) -o lenet main.cu cnnfunc.cu

debug: main.cu cnnfunc.cu 
	nvcc $(CFLAGS) --define-macro D -o lenet main.cu cnnfunc.cu
	./lenet a > debug.log

clean:
	rm -f lenet

.PHONY: clean

