
/* header.h */

#ifndef HEADER_H
#define HEADER_H

__host__ void print_params(const char* name, float* array, int size);
__host__ void print_all_params(float* array, int size);

__host__ void read_params(const char* path, float* array, int size);
__host__ void write_params(const char* path, float* array, int size);
__host__ void check_params(const char* path, float* array, int size);

__host__ void read_binary(const char* path, float* array, int size);
__host__ void write_binary(const char* path, float* array, int size);
__host__ void check_binary(const char* path, float *array, int size);

__host__ void padding(float* input, int isize, int ichan,
                      float* output, int pad);

__host__ void convolution(float* input, int isize, int ichan,
                          float* output, int osize, int ochan,
                          float* weight, float* bias,
                          int ksize, int stride);

__host__ void maxpooling(float* input, int isize, int ichan,
                         float* output, int osize,
                         int ksize, int stride);

__host__ void relu(float* input, int isize, int ichan);

__host__ void lrn(float* input, int isize, int ichan,
                  float* output, int k, int n,
                  float alpha, float beta);

__host__ void classifier(float* input, int isize,
                         float* output, int osize,
                         float* weight, float* bias);

__host__ void softmax(float* input, int isize);

__host__ void show_result(const char* path, float* softmax, int size);
__host__ void norm_image(float* image, int size); 
__host__ void show_image(float* normed_image, int size);

#endif /* HEADER_H */

