
/* lenet.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "header.h"

#define IMAGE_FILE       "./txt/image1000/"
#define CHECK_PARAMS    (0)

#define IMAGE_SIZE      (1 * 28 * 28)

#define CONV1_W_SIZE    (20 * 1 * 5 * 5)
#define CONV1_B_SIZE    (20)
#define CONV1_OUT_SIZE  (20 * 24 * 24)

#define POOL1_OUT_SIZE  (20 * 12 * 12)

#define CONV2_W_SIZE    (50 * 20 * 5 * 5)
#define CONV2_B_SIZE    (50)
#define CONV2_OUT_SIZE  (50 * 8 * 8)

#define POOL2_OUT_SIZE  (50 * 4 * 4)

#define FC1_W_SIZE      (500 * 800)
#define FC1_B_SIZE      (500)
#define FC1_OUT_SIZE    (500)

#define FC2_W_SIZE      (10 * 500)
#define FC2_B_SIZE      (10)
#define FC2_OUT_SIZE    (10)

#define CUDA_SAFE_CALL(func)                                                \
    do {                                                                    \
        cudaError_t err = (func);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n",  \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);      \
            exit(err);                                                      \
        }                                                                   \
    } while (0)

int main()
{
    int imageCount = 0;
    char imageFileName[64];
    char s[32];

    float* image;
    float* conv1_w;
    float* conv1_b;
    float* conv1_out;
    float* pool1_out;
  
    float* conv2_w;
    float* conv2_b;
    float* conv2_out;
    float* pool2_out;

    float* fc1_w;
    float* fc1_b;
    float* fc1_out;

    float* fc2_w;
    float* fc2_b;
    float* fc2_out;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    float elapsedTime;

    cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);

    printf("/// LeNet ///\n");
    fflush(stdout);
  
    printf("Memory allocation ...\n");
    fflush(stdout);

    image = (float*)malloc(sizeof(float) * IMAGE_SIZE);

    conv1_w = (float*)malloc(sizeof(float) * CONV1_W_SIZE);
    conv1_b = (float*)malloc(sizeof(float) * CONV1_B_SIZE);
    conv1_out = (float*)malloc(sizeof(float) * CONV1_OUT_SIZE);
    pool1_out = (float*)malloc(sizeof(float) * POOL1_OUT_SIZE);
    
    conv2_w = (float*)malloc(sizeof(float) * CONV2_W_SIZE);
    conv2_b = (float*)malloc(sizeof(float) * CONV2_B_SIZE);
    conv2_out = (float*)malloc(sizeof(float) * CONV2_OUT_SIZE);
    pool2_out = (float*)malloc(sizeof(float) * POOL2_OUT_SIZE);

    fc1_w = (float*)malloc(sizeof(float) * FC1_W_SIZE);
    fc1_b = (float*)malloc(sizeof(float) * FC1_B_SIZE);
    fc1_out = (float*)malloc(sizeof(float) * FC1_OUT_SIZE);

    fc2_w = (float*)malloc(sizeof(float) * FC2_W_SIZE);
    fc2_b = (float*)malloc(sizeof(float) * FC2_B_SIZE);
    fc2_out = (float*)malloc(sizeof(float) * FC2_OUT_SIZE);
    
    printf("Reading params ...\n");

    /* Print input image values */
    print_params("IMAGE", image, IMAGE_SIZE);

    /* Read Conv1 layer parameters */
    read_params("./txt/conv1_w.txt", conv1_w, CONV1_W_SIZE);
    print_params("CONV1_W", conv1_w, CONV1_W_SIZE);
    read_params("./txt/conv1_b.txt", conv1_b, CONV1_B_SIZE);
    print_params("CONV1_B", conv1_b, CONV1_B_SIZE);
    
    /* Read Conv2 layer parameters */
    read_params("./txt/conv2_w.txt", conv2_w, CONV2_W_SIZE);
    print_params("CONV2_W", conv2_w, CONV2_W_SIZE);
    read_params("./txt/conv2_b.txt", conv2_b, CONV2_B_SIZE);
    print_params("CONV2_B", conv2_b, CONV2_B_SIZE);
    
    /* Read Fc1 layer parameters */
    read_params("./txt/fc1_w.txt", fc1_w, FC1_W_SIZE);
    print_params("FC1_W", fc1_w, FC1_W_SIZE);
    read_params("./txt/fc1_b.txt", fc1_b, FC1_B_SIZE);
    print_params("FC1_B", fc1_b, FC1_B_SIZE);
    
    /* Read Fc2 layer parameters */
    read_params("./txt/fc2_w.txt", fc2_w, FC2_W_SIZE);
    print_params("FC2_W", fc2_w, FC2_W_SIZE);
    read_params("./txt/fc2_b.txt", fc2_b, FC2_B_SIZE);
    print_params("FC2_B", fc2_b, FC2_B_SIZE);

    printf("\n");

    while (1) {
        sprintf(imageFileName, "%simage%03d.txt", IMAGE_FILE, imageCount);
        printf("file: %s\n", imageFileName);
        fflush(stdout);

        read_params(imageFileName, image, IMAGE_SIZE);
        norm_image(image, IMAGE_SIZE);
        
        /* Show image */
        show_image(image, 28);

        printf("\n");
        
        /* Feed-forward */
        printf("Feed forward ...\n");
        fflush(stdout);

        cudaEventRecord(startEvent, 0);

        convolution(image, 28, 1, conv1_out, 24, 20, conv1_w, conv1_b, 5, 1);
        maxpooling(conv1_out, 24, 20, pool1_out, 12, 2, 2);
        convolution(pool1_out, 12, 20, conv2_out, 8, 50, conv2_w, conv2_b, 5, 1);
        maxpooling(conv2_out, 8, 50, pool2_out, 4, 2, 2);
        classifier(pool2_out, 800, fc1_out, 500, fc1_w, fc1_b);
        relu(fc1_out, 1, 500);
        classifier(fc1_out, 500, fc2_out, 10, fc2_w, fc2_b);
        softmax(fc2_out, 10);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        
        printf("\n");
        printf("CPU: time: %f ms\n", elapsedTime);
        printf("\n");
        
        /* Print result */
        print_all_params(fc2_out, 10);
        printf("\n");

        ++imageCount;

        if (imageCount == 1000)
            imageCount = 0;

        printf("Write next image? (y or n): ");
        scanf("%s", s);
        printf("\n");

        if (s[0] == 'y')
            continue;
        
        break;
    }

    return EXIT_SUCCESS;
}

