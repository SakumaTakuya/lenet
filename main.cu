
/* lenet.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "header.h"

#define NUMBER_OF_IMAGE 1000
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

void run_all()
{
    char imageFileName[64];
    char s[32];

    float* hImage;

    float* hConv1W;
    float* hConv1B;
    float* hConv1O;

    float* hPool1O;

    float* hConv2W;
    float* hConv2B;
    float* hConv2O;

    float* hPool2O;

    float* hDens1W;
    float* hDens1B;
    float* hDens1O;

    float* hDens2W;
    float* hDens2B;
    float* hDens2O;
    
    float* dImage;

    float* dConv1W;
    float* dConv1B;
    float* dConv1O;

    float* dPool1O;

    float* dConv2W;
    float* dConv2B;
    float* dConv2O;

    float* dPool2O;

    float* dDens1W;
    float* dDens1B;
    float* dDens1O;

    float* dDens2W;
    float* dDens2B;
    float* dDens2O;

    float* dOutput;

    dim3 bConv1Dim(20,  1, 1); 
    dim3 gConv1Dim(28, 28, 1);
    
    dim3 bPool1Dim(12, 12, 1);
    dim3 gPool1Dim( 1,  1, 1);

    dim3 bConv2Dim( 8,  1, 1); 
    dim3 gConv2Dim(12, 12, 1);

    dim3 bPool2Dim(  4, 4, 1);
    dim3 gPool2Dim(  1, 1, 1);

    dim3 bDens1Dim(800, 1, 1); 
    dim3 gDens1Dim(500, 1, 1);

    dim3 bActi1Dim(500, 1, 1);
    dim3 gActi1Dim(  1, 1, 1);

    dim3 bDens2Dim(500, 1, 1); 
    dim3 gDens2Dim( 10, 1, 1);

    dim3 bActi2Dim( 10, 1, 1);
    dim3 gActi2Dim(  1, 1, 1);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    float elapsedTime;
    float hTime = 0.0;
    float dTime = 0.0;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("/// LeNet ///\n");
    fflush(stdout);

    printf("Allocating host memory ...\n");
    fflush(stdout);

    hImage = (float*)malloc(sizeof(float) * IMAGE_SIZE);

    hConv1W = (float*)malloc(sizeof(float) * CONV1_W_SIZE);
    hConv1B = (float*)malloc(sizeof(float) * CONV1_B_SIZE);
    hConv1O = (float*)malloc(sizeof(float) * CONV1_OUT_SIZE);

    hPool1O = (float*)malloc(sizeof(float) * POOL1_OUT_SIZE);
    
    hConv2W = (float*)malloc(sizeof(float) * CONV2_W_SIZE);
    hConv2B = (float*)malloc(sizeof(float) * CONV2_B_SIZE);
    hConv2O = (float*)malloc(sizeof(float) * CONV2_OUT_SIZE);

    hPool2O = (float*)malloc(sizeof(float) * POOL2_OUT_SIZE);

    hDens1W = (float*)malloc(sizeof(float) * FC1_W_SIZE);
    hDens1B = (float*)malloc(sizeof(float) * FC1_B_SIZE);
    hDens1O = (float*)malloc(sizeof(float) * FC1_OUT_SIZE);

    hDens2W = (float*)malloc(sizeof(float) * FC2_W_SIZE);
    hDens2B = (float*)malloc(sizeof(float) * FC2_B_SIZE);
    hDens2O = (float*)malloc(sizeof(float) * FC2_OUT_SIZE);

    dOutput = (float*)malloc(sizeof(float) * FC2_OUT_SIZE);
    
    printf("Reading parameters ...\n");
    
    /* Read Conv1 layer parameters */
    read_params("./txt/conv1_w.txt", hConv1W, CONV1_W_SIZE);
    print_params("CONV1_W", hConv1W, CONV1_W_SIZE);
    read_params("./txt/conv1_b.txt", hConv1B, CONV1_B_SIZE);
    print_params("CONV1_B", hConv1B, CONV1_B_SIZE);
    
    /* Read Conv2 layer parameters */
    read_params("./txt/conv2_w.txt", hConv2W, CONV2_W_SIZE);
    print_params("CONV2_W", hConv1W, CONV2_W_SIZE);
    read_params("./txt/conv2_b.txt", hConv2B, CONV2_B_SIZE);
    print_params("CONV2_B", hConv2B, CONV2_B_SIZE);
    
    /* Read Fc1 layer parameters */
    read_params("./txt/fc1_w.txt", hDens1W, FC1_W_SIZE);
    print_params("FC1_W", hDens1W, FC1_W_SIZE);
    read_params("./txt/fc1_b.txt", hDens1B, FC1_B_SIZE);
    print_params("FC1_B", hDens1B, FC1_B_SIZE);
    
    /* Read Fc2 layer parameters */
    read_params("./txt/fc2_w.txt", hDens2W, FC2_W_SIZE);
    print_params("FC2_W", hDens2W, FC2_W_SIZE);
    read_params("./txt/fc2_b.txt", hDens2B, FC2_B_SIZE);
    print_params("FC2_B", hDens2B, FC2_B_SIZE);
    
    printf("Allocating device memories ...\n");

    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dImage, IMAGE_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dConv1W, CONV1_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dConv1B, CONV1_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dConv1O, CONV1_OUT_SIZE * sizeof(float)));

    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dPool1O, POOL1_OUT_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dConv2W, CONV2_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dConv2B, CONV2_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dConv2O, CONV2_OUT_SIZE * sizeof(float)));

    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dPool2O, POOL2_OUT_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dDens1W, FC1_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dDens1B, FC1_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dDens1O, FC1_OUT_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dDens2W, FC2_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dDens2B, FC2_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dDens2O, FC2_OUT_SIZE * sizeof(float)));

    printf("Transferring data from host to device ...");

    CUDA_SAFE_CALL(
        cudaMemcpy(dConv1W, hConv1W,
                    CONV1_W_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dConv1B, dConv1B,
                    CONV1_B_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(
        cudaMemcpy(dConv2W, dConv1W,
                    CONV2_W_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dConv2B, hConv2B,
                    CONV2_B_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(
        cudaMemcpy(dDens1W, hDens1W,
                    FC1_W_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dDens1B, hDens2B,
                    FC1_B_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(
        cudaMemcpy(dDens2W, hDens1B,
                    FC2_W_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dDens2B, hDens2B,
                    FC2_B_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));

    printf("\n");
   
    for (int imageCount = 0; imageCount < NUMBER_OF_IMAGE; imageCount++) {
        sprintf(imageFileName, "%simage%03d.txt", IMAGE_FILE, imageCount);
        printf("file: %s\n", imageFileName);
        fflush(stdout);

        read_params(imageFileName, hImage, IMAGE_SIZE);
        norm_image(hImage, IMAGE_SIZE);
        
        /* Show image */
        show_image(hImage, 28);

        printf("\n");
        
        /* Feed-forward in CPU */
        printf("Feed forward ...\n");
        fflush(stdout);

        cudaEventRecord(startEvent, 0);

        convolution(
            hImage, 28, 1, hConv1O, 24, 20, hConv1W, hConv1B, 5, 1);
        maxpooling(
            hConv1O, 24, 20, hPool1O, 12, 2, 2);
        convolution(
            hPool1O, 12, 20, hConv2O, 8, 50, hConv2W, hConv2B, 5, 1);
        maxpooling(
            hConv2O, 8, 50, hPool2O, 4, 2, 2);

        classifier(
            hPool2O, 800, hDens1O, 500, hDens1W, hDens1B);
        relu(
            hDens1O, 1, 500);
        classifier(
           hDens1O, 500, hDens2O, 10, hDens2W, hDens2B);
        softmax(
            hDens2O, 10);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

        hTime += elapsedTime; 
        
        /* Feed-forward in CPU */
        CUDA_SAFE_CALL(
            cudaMemcpy(dImage, hImage, IMAGE_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice));

        cudaEventRecord(startEvent, 0);

        conv2D<28, 1, 56, 24, 48, 5, 10><<<gConv1Dim, bConv1Dim>>>(
            dImage, dConv1O, dConv1W, dConv1B);
        maxpool<12, 24><<<gPool1Dim, bPool1Dim>>>(
            dConv1O, dPool1O);
         
        conv2D<12, 1, 24,  8, 16, 5, 10><<<gConv1Dim, bConv1Dim>>>(
            dPool1O, dConv2O, dConv2W, dConv2B);
        maxpool<4, 8><<<gPool2Dim, bPool2Dim>>>(
            dConv2O, dPool2O);

        dense_relu<800><<<gDens1Dim, bDens1Dim>>>(
            dPool2O, dDens1O, dDens1W, dDens1B);

        dense_softmax<500><<<gDens2Dim, bDens2Dim>>>(
            dDens1O, dDens2O, dDens2W, dDens2B);

        CUDA_SAFE_CALL(
            cudaMemcpy(dOutput, dDens2O, 
                        FC2_OUT_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));


        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

        dTime += elapsedTime; 

        /* Print result */
        printf("CPU \n");
        print_all_params(hDens2O, 10);
        printf("\n");

        printf("GPU \n");
        print_all_params(dOutput, 10);
        printf("\n");

        printf("CPU: time: %f ms\n", hTime);
        printf("GPU: time: %f ms\n", dTime);

        ++imageCount;

        printf("Write next image? (y or n): ");
        scanf("%s", s);
        printf("\n");

        if (s[0] == 'y')
            continue;
        
        break;

    }

    /* Free memory*/
    
    free(hImage);

    free(hConv1W);
    free(hConv1B);
    free(hConv1O);
    free(hPool1O);

    free(hConv2W);
    free(hConv2B);
    free(hConv2O);
    free(hPool2O);

    free(hDens1W);
    free(hDens1B);
    free(hDens1O);

    free(hDens2W);
    free(hDens2B);
    free(hDens2O);

    free(dOutput); 
    
    /* Free device memory */

    CUDA_SAFE_CALL(
        cudaFree(dImage));

    CUDA_SAFE_CALL(
        cudaFree(dConv1O));
    CUDA_SAFE_CALL(
        cudaFree(dConv1B));
    CUDA_SAFE_CALL(
        cudaFree(dConv1W));
    CUDA_SAFE_CALL(
        cudaFree(dPool1O));

    CUDA_SAFE_CALL(
        cudaFree(dConv2O));
    CUDA_SAFE_CALL(
        cudaFree(dConv2B));
    CUDA_SAFE_CALL(
        cudaFree(dConv2W));
    CUDA_SAFE_CALL(
        cudaFree(dPool2O));

    CUDA_SAFE_CALL(
        cudaFree(dDens1W));
    CUDA_SAFE_CALL(
        cudaFree(dDens1B));
    CUDA_SAFE_CALL(
        cudaFree(dDens1O));

    CUDA_SAFE_CALL(
        cudaFree(dDens2W));
    CUDA_SAFE_CALL(
        cudaFree(dDens2B));
    CUDA_SAFE_CALL(
        cudaFree(dDens2O));

    /* Reset device */
    CUDA_SAFE_CALL(cudaDeviceReset());

}


void run_only_cpu()
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

//    return EXIT_SUCCESS;
}

int main()
{
    run_all();

    return EXIT_SUCCESS;
}
