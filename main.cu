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

void* deviceMalloc(size_t size) {
    void* ret;
    CUDA_SAFE_CALL(
        cudaMalloc(&ret, size));
    return ret;
}

void* transFromHost(void* data, size_t size) {
    void* ret = deviceMalloc(size);
    CUDA_SAFE_CALL(
        cudaMemcpy(ret, data, size, cudaMemcpyHostToDevice));
    return ret;
}

void* transFromDevice(void* data, size_t size) {
    void* ret = malloc(size);
    CUDA_SAFE_CALL(
        cudaMemcpy(ret, data, size, cudaMemcpyDeviceToHost));
    return ret;
}

template <int InSize,  int InChannels,  int InSize2, 
          int OutSize, int OutSize2,
          int KernelSize, int KernelSize2>
__global__ void conv2D(float* inImg, float* outImg, 
                       float* weight, float* bias)
{
    /*
        gridDim  == (output channel size, 1, 1)
        blockDim == (input size, input size, 1)
    */
    __shared__ float sharedImg[InSize2];

    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y; 
    const unsigned int bx = blockIdx.x;
    const unsigned int pos = tx + InSize * ty; 
    const int diff = (InSize - OutSize) >> 1;
    const bool outOfRange = tx > OutSize+1 || tx < diff || ty > OutSize+1 || ty < diff;

    float sum = 0;
    #pragma unroll
    for (unsigned int ch = 0; ch < InChannels; ch++) {
        __syncthreads();
        sharedImg[pos] = inImg[pos + InSize2 * ch]; 
#ifdef DO
        if (tx == 0 && bx == 0)
            printf("conv share[%d]: %f\n", pos, sharedImg[pos]);
#endif
        __syncthreads();

        if (outOfRange) {
            continue;
        }

        unsigned int kchan = KernelSize2 * ch + KernelSize2 * InChannels * bx;

        #pragma unroll
        for (unsigned int i = 0; i < KernelSize; i++) {
            #pragma unroll
            for (unsigned int j = 0; j < KernelSize; j++) {
                unsigned int kPos = j + KernelSize * i + kchan;
#ifdef DO
                if (tx == 0 && bx == 0) 
                    printf(
                        "conv (%d, %d, %d)\tw[%d] = %f\tb[%d] = %f\ti[%d] = %f\n",
                        tx, ty, bx, 
                        kPos, weight[kPos], 
                        kPos, bias[kPos], 
                        tx-diff+j + InSize * (ty-diff+i), 
                        sharedImg[tx-diff+j + InSize * (ty-diff+i)]);
#endif
                sum += sharedImg[tx-diff+j + InSize * (ty-diff+i)] * weight[kPos];
            }
        }
    }
#ifdef DO
       if (tx == 0 && bx == 0) {
           printf("conv blockDim %d,%d,%d\n",blockDim.x, blockDim.y, blockDim.z);
           printf("conv thread %d,%d,%d\n",threadIdx.x, threadIdx.y, threadIdx.z);
           printf("conv gridDim %d,%d,%d\n",gridDim.x, gridDim.y, gridDim.z);
       }

#endif
    if (outOfRange) {
#ifdef DO
        printf("conv out of (%d,%d,%d)\n",tx, ty, bx);  
#endif
        return;
    }

    outImg[(tx-diff) + (ty-diff) * OutSize + bx * OutSize2] = sum + bias[bx];
#ifdef DO
    if (bx == 19)
    printf("output%d%d%d[%d,%d,%d] %f \n", 
        blockDim.x,
        blockDim.y,
        blockDim.z,
        (tx-diff),(ty-diff),bx,
        sum + bias[bx]);
#endif
#ifdef DO
    if (tx == 0 && bx == 0) 
        printf("conv bias[%d] = %f \n", bx, bias[bx]);
    if (sum < 0.001)
        printf("conv sum[%d] %f \n", 
            (tx-diff) + (ty-diff) * OutSize + bx * OutSize2, sum);
#endif
}


// TODO shared memoryを使用する
template<int InSize, int InSize2, int OutSize, int OutSize2> 
__global__ void maxpool(float* inImg, float* outImg)
{
    /*
        gridDim == (in/output channel size, 1, 1)
        blockDim == (output size / 2, output size / 2, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int tx2 = tx * 2;
    const unsigned int ty2 = ty * 2;
    const unsigned int ch = bx * InSize2;

#ifdef DO
    if (tx == 0 && bx == 0) {
        printf("pool ch:%d\n", ch);
        printf("pool maxpool:\n[%d,%d]%f \n[%d,%d]%f \n[%d,%d]%f \n[%d,%d]%f\n", 
            tx2  , ty2  , inImg[ tx2    + InSize *  ty2    + ch],
            tx2+1, ty2  , inImg[(tx2+1) + InSize *  ty2    + ch],
            tx2  , ty2+1, inImg[ tx2    + InSize * (ty2+1) + ch],
            tx2+1, ty2+1, inImg[(tx2+1) + InSize * (ty2+1) + ch]);
        printf("pool out pos: %d \n", tx + OutSize * ty + bx * OutSize2);
    }
#endif

    outImg[tx + OutSize * ty + bx * OutSize2] = fmaxf(
        fmaxf(inImg[tx2 + InSize *  ty2    + ch], inImg[(tx2+1) + InSize *  ty2    + ch]),
        fmaxf(inImg[tx2 + InSize * (ty2+1) + ch], inImg[(tx2+1) + InSize * (ty2+1) + ch])
    ); 
}


template <int InSize, int OutSize>
__global__ void dense_exp(float* input, float* output, float* weight, float* bias)
{
    /*
        gridDim == (output size, 1, 1)
        blockDim == (input size, 1, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;

    __shared__ float sharedOut[InSize];

    sharedOut[tx] = input[tx] * weight[tx + InSize * bx]; 
    __syncthreads();

    unsigned int j = 0;
    for (unsigned int i = InSize >> 1; i > 0;i >>= 1){
        if (tx < i) {
            sharedOut[tx] += sharedOut[tx + i];
            if (j == 1 && tx == 0) {
                sharedOut[tx] += sharedOut[i << 1];
            }
        }

        __syncthreads();

        
        j = i & 1;
    }
    
    if (tx == 0){
        output[bx] = expf(sharedOut[0] + bias[bx]);
    }
}


template <int InSize, int OutSize>
__global__ void dense_relu(float* input, float* output, float* weight, float* bias)
{
    /*
        gridDim == (output size, 1, 1)
        blockDim == (input size, 1, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;

    __shared__ float sharedOut[InSize];

#ifdef D
    if (tx == 0) printf("input[%d]=%f\nweight[%d,%d]=%f\n", 
        tx, input[tx], tx, bx, weight[tx + InSize * bx]);
#endif 
    sharedOut[tx] = input[tx] * weight[tx + InSize * bx]; 
#ifdef DO
    if ( tx == 0 && bx == 0)
        printf("dense share[%d]:%f\n",tx, sharedOut[tx]);
#endif
    __syncthreads();

    unsigned int j = 0;
    for (unsigned int i = InSize >> 1; i > 0; i >>= 1 ){
#ifdef DO
        if (tx == 0 && bx == 0) printf("i: %d\n", i);
#endif
        if (tx < i) {
#ifdef D
            printf("%dsum[%d]{%f} = shared[%d]{%f}+shared[%d]{%f}\n",
                bx, tx, sharedOut[tx] + sharedOut[tx+i],
                tx, sharedOut[tx], tx+i, sharedOut[tx+i]);
#endif
            sharedOut[tx] += sharedOut[tx + i];

            if (j == 1 && tx == 0) {
#ifdef D
                printf("%dsum[%d]{%f} = shared[%d]{%f}+shared[%d]{%f}\n",
                    bx, tx, sharedOut[tx] + sharedOut[tx+i],
                    tx, sharedOut[tx], tx+i, sharedOut[tx+i]);
#endif
                sharedOut[tx] += sharedOut[i << 1];
            }
        }

        __syncthreads();
       j = i & 1;
    }
    
    if (tx == 0){
        output[bx] = fmaxf(0, sharedOut[0] + bias[bx]);
    }
}

void test_conv()
{
    float himage[] = {1,1,1,1,
                     1,2,1,1,
                     1,1,1,1,
                     1,1,1,1};
                     

    float hweight[] = {0,0,0,
                      0,2,0,
                      0,0,0,
                      
                      0,0,0,
                      0,2,0,
                      0,0,0};

    float hbias[] = {0,1};

    float hout[8] = {0};
    float hmax[2] = {0};

    float* dimage;
    float* dweight;
    float* dbias;
    float* dout;
    float* dmax;

    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dimage, 16 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dweight, 18 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dbias, 2 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dout, 8 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dmax, 2 * sizeof(float)));

    dim3 cgrid(2,1,1);
    dim3 cblock(4,4,1);
    
    dim3 pgrid(2,1,1);
    dim3 pblock(1,1,1);

    CUDA_SAFE_CALL(
        cudaMemcpy(dimage, himage,
                    16 * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dweight, hweight,
                    18 * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dbias, hbias,
                    2 * sizeof(float),
                    cudaMemcpyHostToDevice));

    conv2D<4,1,16,2,4,3,9><<<cgrid, cblock>>>(
        dimage, dout, dweight, dbias);
    maxpool<2,4,1,1><<<pgrid, pblock>>>(
        dout, dmax); 

    CUDA_SAFE_CALL(
        cudaMemcpy(hout, dout, 
                    8 * sizeof(float),
                    cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(
        cudaMemcpy(hmax, dmax, 
                    2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    printf("ans: \n%f %f\n%f %f\n",4.0, 2.0, 2.0, 2.0); 
    printf(       "%f %f\n%f %f\n",5.0, 3.0, 3.0, 3.0); 
    printf("res: \n%f %f\n%f %f\n", hout[0], hout[1], hout[2], hout[3]);
    printf(       "%f %f\n%f %f\n", hout[4], hout[5], hout[6], hout[7]);
    printf("max: \n%f\n", hmax[0]);
    printf(       "%f\n", hmax[1]);
}

void test_dense()
{
    float himage[] = {1,2,1,1,1,1};
    float hweight[] = { 1,1,2,2,1,1,
                        1,1,1,1,1,1};
    float hbias[] = {1, 0};

    float hout[2] = {0};

    float* dimage;
    float* dweight;
    float* dbias;

    float* dout;

    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dimage, 6 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dweight, 12 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dbias, 2 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dout, 2 * sizeof(float)));

    dim3 dgrid(2,1,1);
    dim3 dblock(6,1,1);

    CUDA_SAFE_CALL(
        cudaMemcpy(dimage, himage,
                    6 * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dweight, hweight,
                    12 * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dbias, hbias,
                    2 * sizeof(float),
                    cudaMemcpyHostToDevice));

    dense_relu<6,2><<<dgrid, dblock>>>(
        dimage, dout, dweight, dbias);
    CUDA_SAFE_CALL(
        cudaMemcpy(hout, dout, 
                    2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    printf("ans: %f %f \n", 10.0,7.0 );
    printf("res: %f %f \n", hout[0], hout[1]);

    dense_exp<6,2><<<dgrid, dblock>>>(
        dimage, dout, dweight, dbias);
    CUDA_SAFE_CALL(
        cudaMemcpy(hout, dout, 
                    2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    printf("ans: %f %f \n", expf(10.0), expf(7.0));
    printf("res: %f %f \n", hout[0], hout[1]);
}

void run_all()
{
    char imageFileName[64];
    char s[32];

    /* allocate host variables */
    float* hImage = (float*) malloc(sizeof(float) * IMAGE_SIZE);

    float* hConv1W = (float*) malloc(sizeof(float) * CONV1_W_SIZE);
    float* hConv1B = (float*) malloc(sizeof(float) * CONV1_B_SIZE);
    float* hConv1O = (float*) malloc(sizeof(float) * CONV1_OUT_SIZE);

    float* hPool1O = (float*) malloc(sizeof(float) * POOL1_OUT_SIZE);

    float* hConv2W = (float*) malloc(sizeof(float) * CONV2_W_SIZE);
    float* hConv2B = (float*) malloc(sizeof(float) * CONV2_B_SIZE);
    float* hConv2O = (float*) malloc(sizeof(float) * CONV2_OUT_SIZE);

    float* hPool2O = (float*) malloc(sizeof(float) * POOL2_OUT_SIZE);

    float* hDense1W = (float*) malloc(sizeof(float) * FC1_W_SIZE);
    float* hDense1B = (float*) malloc(sizeof(float) * FC1_B_SIZE);
    float* hDense1O = (float*) malloc(sizeof(float) * FC1_OUT_SIZE);

    float* hDense2W = (float*) malloc(sizeof(float) * FC2_W_SIZE);
    float* hDense2B = (float*) malloc(sizeof(float) * FC2_B_SIZE);

    float* hDense2O = (float*) malloc(sizeof(float) * FC2_OUT_SIZE);
    float* gDense2O = (float*) malloc(sizeof(float) * FC2_OUT_SIZE);

    /* Rread prams*/ 
    print_params("IMAGE", hImage, IMAGE_SIZE);

    read_params("./txt/conv1_w.txt", hConv1W, CONV1_W_SIZE);
    print_params("CONV1_W", hConv1W, CONV1_W_SIZE);
    read_params("./txt/conv1_b.txt", hConv1B, CONV1_B_SIZE);
    print_params("CONV1_B", hConv1B, CONV1_B_SIZE);

    read_params("./txt/conv2_w.txt", hConv2W, CONV2_W_SIZE);
    print_params("CONV2_W", hConv2W, CONV2_W_SIZE);
    read_params("./txt/conv2_b.txt", hConv2B, CONV2_B_SIZE);
    print_params("CONV2_B", hConv2B, CONV2_B_SIZE);

    read_params("./txt/fc1_w.txt", hDense1W, FC1_W_SIZE);
    print_params("FC1_W", hDense1W, FC1_W_SIZE);
    read_params("./txt/fc1_b.txt", hDense1B, FC1_B_SIZE);
    print_params("FC1_B", hDense1B, FC1_B_SIZE);

    read_params("./txt/fc2_w.txt", hDense2W, FC2_W_SIZE);
    print_params("FC2_W", hDense2W, FC2_W_SIZE);
    read_params("./txt/fc2_b.txt", hDense2B, FC2_B_SIZE);
    print_params("FC2_B", hDense2B, FC2_B_SIZE);
    printf("\n");
 
    /* allocate device variables */
    float* dImage = (float*) transFromHost(hImage, sizeof(float) * IMAGE_SIZE);

    float* dConv1W = (float*) transFromHost(hConv1W, sizeof(float) * CONV1_W_SIZE);
    float* dConv1B = (float*) transFromHost(hConv1B, sizeof(float) * CONV1_B_SIZE);
    float* dConv1O = (float*) deviceMalloc(sizeof(float) * CONV1_OUT_SIZE);
    float* dPool1O = (float*) deviceMalloc(sizeof(float) * POOL1_OUT_SIZE);

    float* dConv2W = (float*) transFromHost(hConv2W, sizeof(float) * CONV2_W_SIZE);
    float* dConv2B = (float*) transFromHost(hConv2B, sizeof(float) * CONV2_B_SIZE);
    float* dConv2O = (float*) deviceMalloc(sizeof(float) * CONV2_OUT_SIZE);
    float* dPool2O = (float*) deviceMalloc(sizeof(float) * POOL2_OUT_SIZE);

    float* dDense1W = (float*) transFromHost(hDense1W, sizeof(float) * FC1_W_SIZE);
    float* dDense1B = (float*) transFromHost(hDense1B, sizeof(float) * FC1_B_SIZE);
    float* dDense1O = (float*) deviceMalloc(sizeof(float) * FC1_OUT_SIZE);

    float* dDense2W = (float*) transFromHost(hDense2W, sizeof(float) * FC2_W_SIZE);
    float* dDense2B = (float*) transFromHost(hDense2B, sizeof(float) * FC2_B_SIZE);
    float* dDense2O = (float*) deviceMalloc(sizeof(float) * FC2_OUT_SIZE);

    dim3 conv1Grid(20, 1, 1);
    dim3 conv1Block(28, 28, 1);

    dim3 pool1Grid(20, 1, 1);
    dim3 pool1Block(12, 12, 1);

    dim3 conv2Grid(50, 1, 1);
    dim3 conv2Block(12, 12, 1);

    dim3 pool2Grid(50, 1, 1);
    dim3 pool2Block(4, 4, 1);

    dim3 dense1Grid(500, 1, 1);
    dim3 dense1Block(800, 1, 1);
    
    dim3 dense2Grid(10, 1, 1);
    dim3 dense2Block(500, 1, 1);

    printf("\n");

    printf("/// LeNet ///\n");
    fflush(stdout);
  
    printf("Memory allocation ...\n");
    fflush(stdout);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    float elapsedTime;
    float hTime;
    float dTime;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

   
    int imageCount;
    for(imageCount = 0; imageCount < 1000; imageCount++) {
        sprintf(imageFileName, "%simage%03d.txt", IMAGE_FILE,imageCount); 
        printf("file: %s\n", imageFileName);
        fflush(stdout);

        read_params(imageFileName, hImage, IMAGE_SIZE);
        norm_image(hImage, IMAGE_SIZE);

        show_image(hImage, 28);
        printf("\n");

        printf("feed forward ... \n");
        fflush(stdout);

        cudaEventRecord(startEvent, 0);

        /* Feed-Forward in CPU */
        convolution(hImage, 28, 1, hConv1O, 24, 20, hConv1W, hConv1B, 5, 1);
        maxpooling(hConv1O, 24, 20, hPool1O, 12, 2, 2);
        convolution(hPool1O, 12, 20, hConv2O, 8, 50, hConv2W, hConv2B, 5, 1);
        maxpooling(hConv2O, 8, 50, hPool2O, 4, 2, 2);

        classifier(hPool2O, 800, hDense1O, 500, hDense1W, hDense1B);
        relu(hDense1O, 1, 500);
        classifier(hDense1O, 500, hDense2O, 10, hDense2W, hDense2B);
        softmax(hDense2O, 10);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        hTime += elapsedTime;

        /* Feed-Forward in GPU */
        CUDA_SAFE_CALL(
            cudaMemcpy(dImage, hImage, IMAGE_SIZE * sizeof(float),
            cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(
            cudaMemcpy(hImage, dImage, IMAGE_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost));

        show_image(hImage, 28);

        cudaEventRecord(startEvent, 0);

        conv2D<28,1,784,24,576,5,25><<<conv1Grid, conv1Block>>>(
            dImage, dConv1O, dConv1W, dConv1B);
        maxpool<24,576,12,144><<<pool1Grid, pool1Block>>>(
            dConv1O, dPool1O);
        conv2D<12,20,144,8,64,5,25><<<conv2Grid, conv2Block>>>(
            dPool1O, dConv2O, dConv2W, dConv2B);
        maxpool<8,64,4,16><<<pool2Grid, pool2Block>>>(
            dConv2O, dPool2O);

        dense_relu<800,500><<<dense1Grid, dense1Block>>>(
            dPool2O, dDense1O, dDense1W, dDense1B);
        dense_exp<500,10><<<dense2Grid, dense2Block>>>(
            dDense1O, dDense2O, dDense2W, dDense2B);
#ifdef D
        float* debC1W = (float*) transFromDevice(
            dConv1W, sizeof(float) * CONV1_W_SIZE);
        float* debC1B = (float*) transFromDevice(
            dConv1B, sizeof(float) * CONV1_B_SIZE);
        float* debC1O = (float*) transFromDevice(
            dConv1O, sizeof(float) * CONV1_OUT_SIZE);
        float* debP1O = (float*) transFromDevice(
            dPool1O, sizeof(float) * POOL1_OUT_SIZE);

        float* debC2W = (float*) transFromDevice(
            dConv2W, sizeof(float) * CONV2_W_SIZE);
        float* debC2B = (float*) transFromDevice(
            dConv2B, sizeof(float) * CONV2_B_SIZE);
        float* debC2O = (float*) transFromDevice(
            dConv2O, sizeof(float) * CONV2_OUT_SIZE);
        float* debP2O = (float*) transFromDevice(
            dPool2O, sizeof(float) * POOL2_OUT_SIZE);

        float* debD1W = (float*) transFromDevice(
            dDense1W, sizeof(float) * FC1_W_SIZE);
        float* debD1B = (float*) transFromDevice(
            dDense1B, sizeof(float) * FC1_B_SIZE);
        float* debD1O = (float*) transFromDevice(
            dDense1O, sizeof(float) * FC1_OUT_SIZE);

        float* debD2W = (float*) transFromDevice(
            dDense2W, sizeof(float) * FC2_W_SIZE);
        float* debD2B = (float*) transFromDevice(
            dDense2B, sizeof(float) * FC2_B_SIZE);
        float* debD2O = (float*) transFromDevice(
            dDense2O, sizeof(float) * FC2_OUT_SIZE);

    print_params("CONV1_W H", hConv1W, CONV1_W_SIZE);
    print_params("CONV1_W D", debC1W, CONV1_W_SIZE);
    print_params("CONV1_B H", hConv1B, CONV1_B_SIZE);
    print_params("CONV1_B D", debC1B, CONV1_B_SIZE);
    print_params("CONV1_O H", hConv1O, CONV1_OUT_SIZE);
    print_params("CONV1_O D", debC1O, CONV1_OUT_SIZE);

    print_params("CONV2_W H", hConv2W, CONV2_W_SIZE);
    print_params("CONV2_W D", debC2W, CONV2_W_SIZE);
    print_params("CONV2_B H", hConv2B, CONV2_B_SIZE);
    print_params("CONV2_B D", debC2B, CONV2_B_SIZE);
    print_params("CONV2_O H", hConv2O, CONV2_OUT_SIZE);
    print_params("CONV2_O D", debC2O, CONV2_OUT_SIZE);

    print_params("FC1_W H", hDense1W, FC1_W_SIZE);
    print_params("FC1_W D", debD1W, FC1_W_SIZE);
    print_params("FC1_B H", hDense1B, FC1_B_SIZE);
    print_params("FC1_B D", debD1B, FC1_B_SIZE);
    print_params("FC1_O H", hDense1O, FC1_OUT_SIZE);
    print_params("FC1_O D", debD1O, FC1_OUT_SIZE);

    print_params("FC2_W H", hDense2W, FC2_W_SIZE);
    print_params("FC2_W D", debD2W, FC2_W_SIZE);
    print_params("FC2_B H", hDense2B, FC2_B_SIZE);
    print_params("FC2_B D", debD2B, FC2_B_SIZE);
    print_params("FC2_O H", hDense2O, FC2_OUT_SIZE);
    print_params("FC2_O D", debD2O, FC2_OUT_SIZE);

    printf("\n");
 
#endif


        CUDA_SAFE_CALL(
            cudaMemcpy(gDense2O, dDense2O,
                        FC2_OUT_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

        float sum = 0;
        for (int m = 0; m < FC2_OUT_SIZE; m++) {
            sum += gDense2O[m];
        }
        for(int m = 0; m < FC2_OUT_SIZE; m++) {
            gDense2O[m] /= sum;
        }

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

        dTime += elapsedTime;


        
        printf("\n");
        printf("CPU: time: %f ms \n", hTime / (1+imageCount));
        printf("\n");
        
        print_all_params(hDense2O, 10);
        printf("\n");

        printf("GPU: time: %f ms \n", dTime / (1+imageCount));
        print_all_params(gDense2O, 10);
        printf("\n");

        printf("Write next image? (y or n): ");
        scanf("%s", s);
        printf("\n");

        if (s[0] == 'n')
            break;
    }

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

int main(int argc, char* argv[])
{
    if (argc == 1) return EXIT_SUCCESS;

    switch(*argv[1]){
        case 'a': run_all(); break;
        case 'o': run_only_cpu(); break;
        case 'c': test_conv(); break;
        case 'd': test_dense(); break;
        default: run_all(); break;
    }

    return EXIT_SUCCESS;
}
