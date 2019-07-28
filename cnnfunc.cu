
/* cnnfunc.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "header.h"

/* int max(int a, int b) { return a > b ? a : b; } */
/* int min(int a, int b) { return a < b ? a : b; } */

__host__ void print_params(const char* name, float* array, int size)
{
    int i;

    printf("%s: ", name);

    for (i = 0; i < 3; ++i)
        printf("%f, ", array[i]);

    printf("...");

    for (i = 2; i >= 0; --i)
        printf(", %f", array[size - i - 1]);

    printf("\n");
    fflush(stdout);
}

__host__ void print_all_params(float* array, int size)
{
    int i;

    for (i = 0; i < size; ++i)
        printf("%d: %f\n", i, array[i]);

    fflush(stdout);
}

__host__ void read_params(const char* path, float* array, int size)
{
    int i;
    FILE* fp;

    if ((fp = fopen(path, "r")) == NULL) {
        printf("could not open file \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < size; ++i)
        fscanf(fp, "%f\n", &array[i]);

    fclose(fp);
}

__host__ void write_params(const char* path, float* array, int size)
{
    int i;
    FILE* fp;

    if ((fp = fopen(path, "w")) == NULL) {
        printf("could not write to file \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < size; ++i)
        fprintf(fp, "%f\n", array[i]);

    fclose(fp);
}

__host__ void check_params(const char* path, float* array, int size)
{
    int i;
    int miss = 0;
    float* debug;

    debug = (float*)malloc(sizeof(float) * size);
    read_params(path, debug, size);

    for (i = 0; i < size; i++) {
        if (fabs(array[i] - debug[i]) > 0.01f) {
            printf("%d:%f, %f\n", i, array[i], debug[i]);
            ++miss;
        }
    }

    printf("check_params(): miss: %d\n", miss);
    fflush(stdout);
}

__host__ void read_binary(const char* path, float* array, int size)
{
    FILE* fp;

    if ((fp = fopen(path, "r")) == NULL) {
        printf("could not open file \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    fread(array, sizeof(float), size, fp);
    fclose(fp);
}

__host__ void write_binary(const char* path, float* array, int size)
{
    FILE* fp;

    if ((fp = fopen(path, "w")) == NULL) {
        printf("could not write to file \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    fwrite(array, sizeof(float), size, fp);
    fclose(fp);
}

__host__ void check_binary(const char* path, float* array, int size)
{
    int i;
    int miss = 0;
    float* debug;

    debug = (float*)malloc(sizeof(float) * size);
    read_binary(path, debug, size);

    for (i = 0; i < size; ++i) {
        if (fabs(array[i] - debug[i]) > 0.01f) {
            printf("%d:%f, %f\n", i, array[i], debug[i]);
            ++miss;
        }
    }

    printf("check_binary(): miss: %d\n", miss);
    fflush(stdout);
}

__host__ void padding(float* input, int isize, int ichan,
                      float* output, int pad)
{
    int ocol;
    int orow;
    int och;
    int osize = isize + pad + pad;
    
    for (och = 0; och < ichan; ++och)
        for (orow = 0; orow < osize; ++orow)
            for (ocol = 0; ocol < osize; ++ocol)
                *(output + och * osize * osize + orow * osize + ocol) = 0.0f;

    for (och = 0; och < ichan; ++och)
        for (orow = 0; orow < isize; ++orow)
            for (ocol = 0; ocol < isize; ++ocol)
                *(output + och * osize * osize + (orow + pad) * osize + (ocol + pad)) =
                    *(input + och * isize * isize + orow * isize + ocol);
}

__host__ void convolution(float* input, int isize, int ichan,
                          float* output, int osize, int ochan,
                          float* weight, float* bias,
                          int ksize, int stride)
{
    /*
     * Data format:
     * input[ch (< ichan)][row (< isize)][[col (< isize)]
     * output[ch (< ochan)][row (< osize)][col (< osize)]
     * weight[kernel (< ochan)][ch (< ichan)][row (< ksize)][col (< ksize)]
     * bias[kernel (< ochan)]
     */

    int ocol;
    int orow;
    int och;
    int kcol;
    int krow;
    int kch;
#ifdef D
    printf("convolution(): isize: %d, ichan: %d, osize: %d, "
           "ochan: %d, ksize: %d, stride: %d\n",
           isize, ichan, osize, ochan, ksize, stride);
    fflush(stdout);
#endif

    for (och = 0; och < ochan; ++och) {
        for (orow = 0; orow < osize; ++orow) {
            for (ocol = 0; ocol < osize; ++ocol) {
                *(output + och * osize * osize + orow * osize + ocol) = 0.0f;

                for (krow = 0; krow < ksize; ++krow) {
                    for (kcol = 0; kcol < ksize; ++kcol) {
                        for (kch = 0; kch < ichan; ++kch) {
                            /*
                             * output[och][orow][ocol] +=
                             *     weight[och][kch][krow][kcol] *
                             *     input[kch][orow * stride + krow][ocol * stride + kcol];
                             */
                            *(output + och * osize * osize + orow * osize + ocol) +=
                                *(weight + och * ichan * ksize * ksize
                                  + kch * ksize * ksize + krow * ksize + kcol) *
                                *(input + kch * isize * isize
                                  + (orow * stride + krow) * isize
                                  + (ocol * stride + kcol));
                        }
                    }
                }

                *(output + och * osize * osize + orow * osize + ocol) += *(bias + och);
            }
        }
    }
}

__host__ void maxpooling(float* input, int isize, int ichan,
                         float* output, int osize,
                         int ksize, int stride)
{
    int ocol;
    int orow;
    int och;
    int kcol;
    int krow;
    float max;
    float tmp;
#ifdef D
    printf("maxpooling(): isize: %d, ichan: %d, osize: %d, "
           "ksize: %d, stride: %d\n",
           isize, ichan, osize, ksize, stride);
    fflush(stdout);
#endif

    for (och = 0; och < ichan; ++och) {
        for (orow = 0; orow < osize; ++orow) {
            for (ocol = 0; ocol < osize; ++ocol) {
                max = -256.0f;

                for (krow = 0; krow < ksize; ++krow) {
                    for (kcol = 0; kcol < ksize; ++kcol) {
                        /* tmp = input[och][orow * stride + krow][ocol * stride + kcol]; */
                        tmp = *(input + och * isize * isize
                                + (orow * stride + krow) * isize
                                + (ocol * stride + kcol));

                        if (max < tmp)
                            max = tmp;
                    }
                }

                *(output + och * osize * osize + osize * orow + ocol) = max;
            }
        }
    }
}

__host__ void relu(float* input, int isize, int ichan)
{
    int ocol;
    int orow;
    int och;
#ifdef D
    printf("relu(): isize: %d, ichan: %d\n", isize, ichan);
    fflush(stdout);
#endif
    for (och = 0; och < ichan; ++och)
        for (orow = 0; orow < isize; ++orow)
            for (ocol = 0; ocol < isize; ++ocol)
                if (*(input + och * isize * isize + orow * isize + ocol) < 0.0f)
                    *(input + och * isize * isize + orow * isize + ocol) = 0.0f;
}

__host__ void lrn(float* input, int isize, int ichan,
                  float* output, int k, int n,
                  float alpha, float beta)
{
    /* Local Response Normalization (LRN) */

    int ocol;
    int orow;
    int och;
    int j;
    float sum;
    float tmp;

    alpha = 0.0001f;
    beta = 0.75f;
#ifdef D
    printf("lrn(): isize: %d, ichan: %d, k: %d, n: %d, a: %f, b: %f\n",
           isize, ichan, k, n, alpha, beta);
    fflush(stdout);
#endif
    for (och = 0; och < ichan; ++och) {
        for (orow = 0; orow < isize; ++orow) {
            for (ocol = 0; ocol < isize; ++ocol) {
                sum = 0.0f;

                for (j = max(0, och - (n / 2));
                     j <= min(ichan - 1, och + (n / 2)); ++j) {
                    tmp = *(input + j * isize * isize + orow * isize + ocol);
                    sum += tmp * tmp;
                }
                
                *(output + och * isize * isize + orow * isize + ocol) =
                    *(input + och * isize * isize + orow * isize + ocol) *
                    powf((float)k + alpha / (float)n * sum, (float)-beta);
            }
        }
    }
}

__host__ void classifier(float* input, int isize,
                         float* output, int osize,
                         float* weight, float* bias)
{
    int i;
    int j;
#ifdef D
    printf("classifier(): isize: %d, osize: %d\n", isize, osize);
    fflush(stdout);
#endif
    for (i = 0; i < osize; ++i) {
        *(output + i) = 0.0f;

        for (j = 0; j < isize; ++j)
            /* output[i] += weight[i][j] * input[j]; */
            *(output + i) += *(weight + i * isize + j) * *(input + j);

        *(output + i) += *(bias + i);
    }
}

__host__ void softmax(float* input, int isize)
{
    int i;
    float sum = 0.0f;
#ifdef D
    printf("softmax(): isize: %d\n", isize);
    fflush(stdout);
#endif
    for (i = 0; i < isize; ++i)
        sum += expf(*(input + i));
    
    for (i = 0; i < isize; ++i)
        *(input + i) = expf(*(input + i)) / sum;
}

__host__ void show_result(const char* path, float* softmax, int size)
{
    int i;
    int first = 0;
    int second = 0;
    int third = 0;
    FILE *fp;

    char category[size][64];
    char tmp[64];

    if ((fp = fopen(path, "r")) == NULL) {
        printf("could not open file \'%s\'\n", path);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < size; ++i)
        fscanf(fp, "%s %[^\n]\n", tmp, category[i]);

    fclose(fp);

    printf("Show result: \n");

    for (i = 0; i < size; ++i) {
        if (softmax[i] > softmax[third]) {
            third = i;
            if (softmax[i] > softmax[second]) {
                third = second;
                second = i;
                if (softmax[i] > softmax[first]) {
                    second = first;
                    first = i;
                }
            }
        }
    }

    printf("%s: %f\n", category[first], softmax[first] * 100.0f);
    printf("%s: %f\n", category[second], softmax[second] * 100.0f);
    printf("%s: %f\n", category[third], softmax[third] * 100.0f);
    fflush(stdout);
}

__host__ void norm_image(float* image, int size)
{
    int i;

    for (i = 0; i < size; ++i)
        image[i] /= 255.0f;
}

__host__ void show_image(float* normed_image, int size)
{
    int i;
    int j;
    
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            if (normed_image[i * size + j] > 0.5f)
                printf("* ");
            else
                printf("  ");
        }

        printf("\n");
    }
}

