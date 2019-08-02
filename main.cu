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

void* deviceMalloc(size_t size) 
{
    void* ret;
    CUDA_SAFE_CALL(
        cudaMalloc(&ret, size));
    return ret;
}

void* transFromHost(void* data, size_t size)
{
    void* ret = deviceMalloc(size);
    CUDA_SAFE_CALL(
        cudaMemcpy(ret, data, size, cudaMemcpyHostToDevice));
    return ret;
}

void deviceFree(void* data)
{
    CUDA_SAFE_CALL(cudaFree(data));
}

void* transFromDevice(void* data, size_t size) 
{
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
    const bool outOfRange = tx>OutSize+1 || tx<diff || ty>OutSize+1 || ty<diff;

    float sum = 0;
    #pragma unroll
    for (unsigned int ch = 0; ch < InChannels; ch++) {
        __syncthreads();
        sharedImg[pos] = inImg[pos + InSize2 * ch]; 
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
                sum += sharedImg[tx-diff+j + InSize * (ty-diff+i)] * weight[kPos];
            }
        }
    }

    if (outOfRange) {
        return;
    }

    outImg[(tx-diff) + (ty-diff) * OutSize + bx * OutSize2] = sum + bias[bx];
}


template <int InSize,  int InChannels,  int InSize2, 
          int OutSize, int OutSize2, int OutSizeHalf, int OutSizeHalf2,
          int KernelSize, int KernelSize2, int Diff >
__global__ void conv2D_pool_vanilla(float* inImg, float* outImg, 
                       float* weight, float* bias)
{
    /*
        gridDim  == (output channel size, 1, 1)
        blockDim == (input size, input size, 1)
    */

    __shared__ float sharedImg[InSize2];
    __shared__ float sharedOut[OutSize2]; 

    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y; 
    const unsigned int bx = blockIdx.x;
    const unsigned int outx = tx-Diff;
    const unsigned int outy = ty-Diff;
    const unsigned int pos = tx + InSize * ty; 
    const unsigned int kerChPos = KernelSize2 * InChannels * bx;
    const unsigned int inSizOuty = InSize * outy;
    const bool outOfRange = tx>OutSize+1 || tx<Diff || ty>OutSize+1 || ty<Diff;

    unsigned int imgCh = 0;
    unsigned int kchan = kerChPos;
    float sum = 0;
    #pragma unroll
    for (unsigned int ch = 0; ch < InChannels; ch++) {
        __syncthreads();
        sharedImg[pos] = inImg[pos + imgCh]; 
        __syncthreads();

        if (outOfRange) {
            continue;
        }

        unsigned int kPos = kchan;
        unsigned int imgyPos = inSizOuty;
        #pragma unroll
        for (unsigned int i = 0; i < KernelSize; i++) {
            #pragma unroll
            for (unsigned int j = 0; j < KernelSize; j++) {
                sum += sharedImg[outx+j + imgyPos] * weight[kPos];
                kPos++;
            }
            imgyPos += InSize;
        }
        imgCh += InSize2;
        kchan += KernelSize2;
    }

    if (outOfRange) {
        return;
    }
     
    const unsigned int outyPos = outy * OutSize;

    sharedOut[outx + outyPos] = sum + bias[bx];
    __syncthreads();

    if (outx >= OutSizeHalf || outy >= OutSizeHalf) {
        return;
    }

    const int outx2  = outx << 1;
    const int outy2  = (outy << 1) * OutSize;
    const int outx2Nex = outx2 + 1;
    const int outy2Nex = ((outy << 1) + 1) * OutSize;

    outImg[outx + (outyPos >> 1) + bx * OutSizeHalf2] = 
        fmaxf(
            fmaxf(sharedOut[outx2    + outy2], 
                  sharedOut[outx2Nex + outy2]),
            fmaxf(sharedOut[outx2    + outy2Nex], 
                  sharedOut[outx2Nex + outy2Nex])
        );
}


template <int InSize,  int InChannels,  int InSize2, 
          int OutSize, int OutSize2, int OutSizeHalf, int OutSizeHalf2,
          int KernelSize, int KernelSize2, int Diff, int BaseSize, int BaseSize2 >
__global__ void conv2D_pool_quad(float* inImg, float* outImg, 
                       float* weight, float* bias)
{
    /*
        gridDim  == (output channel size, 2, 2)
        blockDim == (input size/2+2, input size/2+2, 1)
    */

    __shared__ float sharedImg[InSize2];
    __shared__ float sharedOut[OutSize2]; 

    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y; 
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    const unsigned int bz = blockIdx.z;
    const unsigned int outx = tx-Diff;
    const unsigned int outy = ty-Diff;
    const unsigned int pos = tx + InSize * ty; 
    const unsigned int kerChPos = KernelSize2 * InChannels * bx;
    const unsigned int inSizOuty = InSize * outy;
    const bool outOfRange = tx >= InSize - Diff ||
                            tx <  Diff          || 
                            ty >= InSize - Diff || 
                            ty <  Diff;

    unsigned int imgCh = 0;
    unsigned int kchan = kerChPos;
    float sum = 0;
    #pragma unroll
    for (unsigned int ch = 0; ch < InChannels; ch++) {
        __syncthreads();
        sharedImg[pos] = inImg[(tx + (InSize-Diff) * by) + (ty + (InSize-Diff) * bz) * BaseSize + imgCh]; 
        __syncthreads();

        if (outOfRange) {
            continue;
        }

        unsigned int kPos = kchan;
        unsigned int imgyPos = inSizOuty;
        #pragma unroll
        for (unsigned int i = 0; i < KernelSize; i++) {
            #pragma unroll
            for (unsigned int j = 0; j < KernelSize; j++) {
                sum += sharedImg[outx+j + imgyPos] * weight[kPos];
                kPos++;
            }
            imgyPos += InSize;
        }
        imgCh += BaseSize2;
        kchan += KernelSize2;
    }

    if (outOfRange) {
        return;
    }
     
    const unsigned int outyPos = outy * OutSize;

    sharedOut[outx + outyPos] = sum + bias[bx];
    __syncthreads();

    if (outx >= OutSizeHalf || outy >= OutSizeHalf) {
        return;
    }

    const int outx2  = outx << 1;
    const int outy2  = (outy << 1) * OutSize;
    const int outx2Nex = outx2 + 1;
    const int outy2Nex = ((outy << 1) + 1) * OutSize;

    outImg[outx + (outyPos >> 1) + bx * OutSizeHalf2] = 
        fmaxf(
            fmaxf(sharedOut[outx2    + outy2], 
                  sharedOut[outx2Nex + outy2]),
            fmaxf(sharedOut[outx2    + outy2Nex], 
                  sharedOut[outx2Nex + outy2Nex])
        );
}


template <int InSize,  int InChannels,  int InSize2, 
          int OutSize, int OutSize2, int OutSizeHalf, int OutSizeHalf2,
          int KernelSize, int KernelSize2, int Diff, int KernelScale>
__global__ void conv2D_pool_ch1(float* inImg, float* outImg, 
                       float* weight, float* bias)
{
    /*
        gridDim  == (output channel size, 1, 1)
        blockDim == (input size, input size, Kernel)
    */

    __shared__ float sharedImg[InSize2];
    __shared__ float sharedOut[OutSize2]; 
    __shared__ float sharedWeight[KernelSize2];

    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y; 
    const unsigned int bx = blockIdx.x;
    const unsigned int outx = tx-Diff;
    const unsigned int outy = ty-Diff;
    const unsigned int pos = tx + InSize * ty; 
    const unsigned int kChPos = KernelScale * bx;
    const unsigned int inSizOuty = InSize * outy;
    const bool outOfRange = tx >= InSize-Diff || 
                            tx <  Diff        || 
                            ty >= InSize-Diff || 
                            ty <  Diff;

    if(tx < KernelSize2) {
        sharedWeight[tx] = weight[tx + kChPos];
    }

    float sum = 0;
    __syncthreads();

    sharedImg[pos] = inImg[pos]; 
    __syncthreads();

    if (outOfRange) {
        return;
    }

    unsigned int kPos = 0;
    unsigned int imgyPos = inSizOuty;
    #pragma unroll
    for (unsigned int i = 0; i < KernelSize; i++) {
        #pragma unroll
        for (unsigned int j = 0; j < KernelSize; j++) {
            sum += sharedImg[outx+j + imgyPos] * sharedWeight[kPos];
            kPos++;
        }
        imgyPos += InSize;
    }

    //if (outOfRange) {
    //    return;
    //}
     
    const unsigned int outyPos = outy * OutSize;

    sharedOut[outx + outyPos] = sum + bias[bx];
    __syncthreads();

    if (outx >= OutSizeHalf || outy >= OutSizeHalf) {
        return;
    }

    const int outx2  = outx << 1;
    const int outy2  = (outy << 1) * OutSize;
    const int outx2Nex = outx2 + 1;
    const int outy2Nex = ((outy << 1) + 1) * OutSize;

    outImg[outx + (outyPos >> 1) + bx * OutSizeHalf2] = 
        fmaxf(
            fmaxf(sharedOut[outx2    + outy2], 
                  sharedOut[outx2Nex + outy2]),
            fmaxf(sharedOut[outx2    + outy2Nex], 
                  sharedOut[outx2Nex + outy2Nex])
        );
}


/*
    Depth == BlockDim.z
    InSize2Depth == InSize2 << (Depth >> 1)
    KernelSize2Depth == KernelSize2 << (Depth >> 1)
    // 一つ一つのサイズはKernelSize * KernelSize * InChannels
    KernelScale = KernelSize2 * InChannels
    LoopCount == InChannels >> (Depth >> 1)
*/
template <int InSize,  int InChannels,  int InSize2, 
          int OutSize, int OutSize2, int OutSizeHalf, int OutSizeHalf2,
          int KernelSize, int KernelSize2, int Diff, 
          int Depth, int InSize2Depth, int KernelSize2Depth,
          int KernelScale, int LoopCount>
__global__ void conv2D_pool(float* inImg, float* outImg, 
                            float* weight, float* bias)
{
    /*
        gridDim  == (output channel size, 1, 1)
        blockDim == (input size, input size, Depth)
    */

    __shared__ float sharedImg[InSize2][Depth];
    __shared__ float sharedOut[OutSize2][Depth]; 

    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y; 
    const unsigned int tz = threadIdx.z; 
    const unsigned int bx = blockIdx.x;
    const unsigned int imgTar = InSize2 * tz;
    const unsigned int kerTar = KernelSize2 * tz;

    const unsigned int outx = tx - Diff;
    const unsigned int outy = ty - Diff;

    const unsigned int pos = tx + InSize * ty; 
    const unsigned int inSizOuty = InSize * outy;
    const bool outOfRange = tx >= InSize - Diff ||
                            tx <  Diff          || 
                            ty >= InSize - Diff || 
                            ty <  Diff;

    unsigned int imgCh = imgTar;
    unsigned int kchan = kerTar + KernelScale * bx;
    float sum = 0;
    #pragma unroll
    for (unsigned int ch = 0; ch < LoopCount; ch++) {
        __syncthreads();
        sharedImg[pos][tz] = inImg[pos + imgCh]; 
        //printf("share[%d][%d]=%f\tinimg[%d]=%f\n", 
        //    pos, tz, sharedImg[pos][tz], pos+imgCh, inImg[pos + imgCh]);
        __syncthreads();

        if (outOfRange) {
            continue;
        }

        unsigned int kPos = kchan;
        unsigned int imgyPos = inSizOuty;
        #pragma unroll
        for (unsigned int i = 0; i < KernelSize; i++) {
            #pragma unroll
            for (unsigned int j = 0; j < KernelSize; j++) {
   
                //printf("[%d,%d]%d,%d,%d,%d\tshrePos=%d kPos=%d\nw=%f\ti=%f\n",
                //       i, j, tx, ty, tz, bx, 
                //       outx+j + imgyPos, kPos, 
                //       weight[kPos], sharedImg[outx+j + imgyPos][tz]);
               
                sum += sharedImg[outx+j + imgyPos][tz] * weight[kPos];
                kPos++;
            }
            imgyPos += InSize;
        }
        imgCh += InSize2Depth;
        kchan += KernelSize2Depth;
    }

    if (outOfRange) {
        return;
    }

    const unsigned int outyPos = outy * OutSize;
    const unsigned int outPos = outx + outyPos; 

    sharedOut[outPos][tz] = sum + (tz == 0 ? bias[bx] : 0);
    //printf("(%d,%d,%d)shared[%d][%d]=%f\n", tx,ty,bx,outPos, tz,sharedOut[outPos][tz] );
    __syncthreads();

    if (tz != 0) {
        return;
    }

    #pragma unroll
    for (unsigned int i = 1; i < Depth; i++){
        //printf("add(%d):%f+%f\n", 
        //    outPos, sharedOut[outPos][0], sharedOut[outPos][i]);
        sharedOut[outPos][0] += sharedOut[outPos][i]; 
    }
    
    __syncthreads();
    
    if (outx >= OutSizeHalf || outy >= OutSizeHalf) {
        return;
    }

    const int outx2  = outx << 1;
    const int outy2  = (outy << 1) * OutSize;
    const int outx2Nex = outx2 + 1;
    const int outy2Nex = ((outy << 1) + 1) * OutSize;

    outImg[outx + (outyPos >> 1) + bx * OutSizeHalf2] = 
        fmaxf(
            fmaxf(sharedOut[outx2    + outy2][0], 
                  sharedOut[outx2Nex + outy2][0]),
            fmaxf(sharedOut[outx2    + outy2Nex][0], 
                  sharedOut[outx2Nex + outy2Nex][0])
        );
}


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
    const unsigned int tx2 = tx << 1;
    const unsigned int ty2 = ty << 1;
    const unsigned int ch = bx * InSize2;
    
    outImg[tx + OutSize * ty + bx * OutSize2] = fmaxf(
        fmaxf(inImg[tx2     + InSize *  ty2    + ch], 
              inImg[(tx2+1) + InSize *  ty2    + ch]),
        fmaxf(inImg[tx2     + InSize * (ty2+1) + ch], 
              inImg[(tx2+1) + InSize * (ty2+1) + ch])
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

#ifdef DO
    if (tx == 0) printf("input[%d]=%f\nweight[%d,%d]=%f\n", 
        tx, input[tx], tx, bx, weight[tx + InSize * bx]);
#endif 

    sharedOut[tx] = input[tx] * weight[tx + InSize * bx]; 
    __syncthreads();

    unsigned int j = 0;
    for (unsigned int i = InSize >> 1; i > 0; i >>= 1 ){
        if (tx < i) {
#ifdef DO
            printf("%dsum[%d]{%f} = shared[%d]{%f}+shared[%d]{%f}\n",
                bx, tx, sharedOut[tx] + sharedOut[tx+i],
                tx, sharedOut[tx], tx+i, sharedOut[tx+i]);
#endif
            sharedOut[tx] += sharedOut[tx + i];

            if (j == 1 && tx == 0) {
#ifdef DO
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


template <int InSize, int OutSize>
__global__ void dense_relu_half(float* input, float* output, float* weight, float* bias)
{
    /*
        gridDim == (output size, 1, 1)
        blockDim == (input size / 2, 1, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;

    __shared__ float sharedOut[InSize];
    sharedOut[tx] = input[tx         ] * weight[tx          + (InSize << 1) * bx] + 
                    input[tx + InSize] * weight[tx + InSize + (InSize << 1) * bx]; 

    //printf("(%d)out[%d]=%f*%f+%f*%f\n",bx, tx ,input[tx] ,weight[tx+ (InSize << 1) * bx], 
    //                input[tx + InSize], weight[tx + InSize + (InSize << 1) * bx]); 
    __syncthreads();

    unsigned int j = (InSize >> 1) & 1;
    for (unsigned int i = InSize >> 1; i > 0; i >>= 1){
        if (tx < i) {
//            printf("(%d) out[%d]=(%d)%f+(%d)%f\n",bx,tx,tx,sharedOut[tx], tx+i,sharedOut[tx + i]);
            sharedOut[tx] += sharedOut[tx + i];

            if (j == 1 && tx == 0) {
//                printf("(%d) out[%d]=(%d)%f+(%d)%f\n",bx,tx,tx, sharedOut[tx],i<<1, sharedOut[i << 1]);
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


template <int InSize, int OutSize, int InSizeD, int InSizeT>
__global__ void dense_relu_quarter(float* input, float* output, float* weight, float* bias)
{
    /*
        gridDim == (output size, 1, 1)
        blockDim == (input size / 2, 1, 1)
    */

    const unsigned int inWidth = InSize << 2;
    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;

    __shared__ float sharedOut[InSize];
    sharedOut[tx] = input[tx          ] * weight[tx           + inWidth * bx] + 
                    input[tx + InSize ] * weight[tx + InSize  + inWidth * bx] + 
                    input[tx + InSizeD] * weight[tx + InSizeD + inWidth * bx] + 
                    input[tx + InSizeT] * weight[tx + InSizeT + inWidth * bx]; 

    //printf("(%d)out[%d]=%f*%f+%f*%f\n",bx, tx ,input[tx] ,weight[tx+ (InSize << 1) * bx], 
    //                input[tx + InSize], weight[tx + InSize + (InSize << 1) * bx]); 
    __syncthreads();

    unsigned int j = (InSize >> 1) & 1;
    for (unsigned int i = InSize >> 1; i > 0; i >>= 1){
        if (tx < i) {
//            printf("(%d) out[%d]=(%d)%f+(%d)%f\n",bx,tx,tx,sharedOut[tx], tx+i,sharedOut[tx + i]);
            sharedOut[tx] += sharedOut[tx + i];

            if (j == 1 && tx == 0) {
//                printf("(%d) out[%d]=(%d)%f+(%d)%f\n",bx,tx,tx, sharedOut[tx],i<<1, sharedOut[i << 1]);
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


template <int InSize, int OutSize, int InSizeD, int InSizeT, int InSizeQ, int InSizeF, int InSizeS, int InSizeSe>
__global__ void dense_relu_eighth(float* input, float* output, float* weight, float* bias)
{
    /*
        gridDim == (output size, 1, 1)
        blockDim == (input size / 8, 1, 1)
    */

    const unsigned int inWidth = InSize << 3;
    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;

    __shared__ float sharedOut[InSize];
    sharedOut[tx] = input[tx           ] * weight[tx            + inWidth * bx] + 
                    input[tx + InSize  ] * weight[tx + InSize   + inWidth * bx] + 
                    input[tx + InSizeD ] * weight[tx + InSizeD  + inWidth * bx] + 
                    input[tx + InSizeT ] * weight[tx + InSizeT  + inWidth * bx] + 
                    input[tx + InSizeQ ] * weight[tx + InSizeQ  + inWidth * bx] + 
                    input[tx + InSizeF ] * weight[tx + InSizeF  + inWidth * bx] + 
                    input[tx + InSizeS ] * weight[tx + InSizeS  + inWidth * bx] + 
                    input[tx + InSizeSe] * weight[tx + InSizeSe + inWidth * bx]; 

    //printf("(%d)out[%d]=%f*%f+%f*%f\n",bx, tx ,input[tx] ,weight[tx+ (InSize << 1) * bx], 
    //                input[tx + InSize], weight[tx + InSize + (InSize << 1) * bx]); 
    __syncthreads();

    unsigned int j = (InSize >> 1) & 1;
    for (unsigned int i = InSize >> 1; i > 0; i >>= 1){
        if (tx < i) {
//            printf("(%d) out[%d]=(%d)%f+(%d)%f\n",bx,tx,tx,sharedOut[tx], tx+i,sharedOut[tx + i]);
            sharedOut[tx] += sharedOut[tx + i];

            if (j == 1 && tx == 0) {
//                printf("(%d) out[%d]=(%d)%f+(%d)%f\n",bx,tx,tx, sharedOut[tx],i<<1, sharedOut[i << 1]);
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
                     1,1,1,1,
                     
                     1,1,1,1,
                     1,2,1,1,
                     1,1,1,1,
                     1,1,1,1};
                     

    float hweight[] = {0,0,0,
                      0,2,0,
                      0,0,0,
                      
                      0,0,0,
                      0,2,0,
                      0,0,0,

                      0,0,0,
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
        cudaMalloc((void**)&dimage, 32 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dweight, 36 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dbias, 2 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dout, 8 * sizeof(float)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&dmax, 2 * sizeof(float)));

    dim3 cgrid(2,1,1);
    dim3 cblock(4,4,2);
    
    dim3 pgrid(2,1,1);
    dim3 pblock(1,1,1);

    CUDA_SAFE_CALL(
        cudaMemcpy(dimage, himage,
                    32 * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dweight, hweight,
                    36 * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(dbias, hbias,
                    2 * sizeof(float),
                    cudaMemcpyHostToDevice));

    conv2D_pool<4,2,16,2,4,1,1,3,9,1,2,32,18,18,1>
        <<<cgrid, cblock>>>(dimage, dmax, dweight, dbias);

    CUDA_SAFE_CALL(
        cudaMemcpy(hout, dout, 
                    8 * sizeof(float),
                    cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(
        cudaMemcpy(hmax, dmax, 
                    2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    //printf("ans: \n%f %f\n%f %f\n",4.0, 2.0, 2.0, 2.0); 
    //printf(       "%f %f\n%f %f\n",5.0, 3.0, 3.0, 3.0); 
    //printf("res: \n%f %f\n%f %f\n", hout[0], hout[1], hout[2], hout[3]);
    //printf(       "%f %f\n%f %f\n", hout[4], hout[5], hout[6], hout[7]);
    printf("max: \n%f\n", hmax[0]);
    printf(       "%f\n", hmax[1]);
}

void test_dense()
{
    float himage[]  = {1,2,1,1,1,1};
    float hweight[] = {1,1,2,2,1,1,
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

    dim3 dblock_h(3,1,1);

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

    dense_relu_half<3,2><<<dgrid, dblock_h>>>(
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
    dim3 conv2Block(12, 12, 4);

    dim3 pool2Grid(50, 1, 1);
    dim3 pool2Block(4, 4, 1);

    dim3 dense1Grid(500, 1, 1);
    dim3 dense1Block(200, 1, 1);
    
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
    float hTime = 0.0f;
    float dTime = 0.0f;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

   
    int imageCount;
    for(imageCount = 0; imageCount < 1000; imageCount++) {
        sprintf(imageFileName, "%simage%03d.txt", IMAGE_FILE,imageCount); 
#ifdef D
        printf("file: %s\n", imageFileName);
        fflush(stdout);
#endif
        read_params(imageFileName, hImage, IMAGE_SIZE);
        norm_image(hImage, IMAGE_SIZE);
#ifdef D
        show_image(hImage, 28);
        printf("\n");

        printf("feed forward ... \n");
        fflush(stdout);
#endif
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

        cudaEventRecord(startEvent, 0);
/*
        conv2D<28,1,784,24,576,5,25><<<conv1Grid, conv1Block>>>(
            dImage, dConv1O, dConv1W, dConv1B);
        maxpool<24,576,12,144><<<pool1Grid, pool1Block>>>(
            dConv1O, dPool1O);
        conv2D<12,20,144,8,64,5,25><<<conv2Grid, conv2Block>>>(
            dPool1O, dConv2O, dConv2W, dConv2B);
        maxpool<8,64,4,16><<<pool2Grid, pool2Block>>>(
            dConv2O, dPool2O);
*/

        conv2D_pool_ch1<28, 1,784,24,576,12,144,5,25,2,25>
            <<<conv1Grid,conv1Block>>>(dImage, dPool1O, dConv1W, dConv1B);
        //conv2D_pool<12,20,144, 8, 64, 4, 16,5,25,2,4,288, 50,500,10>
        //    <<<conv2Grid, conv2Block>>>(dPool1O, dPool2O, dConv2W, dConv2B);
        conv2D_pool<12,20,144, 8, 64, 4, 16,5,25,2,4,576,100,500,5>
            <<<conv2Grid, conv2Block>>>(dPool1O, dPool2O, dConv2W, dConv2B);

        //dense_relu_eighth<100,500,200,300,400,500,600,700><<<dense1Grid, dense1Block>>>(
        //    dPool2O, dDense1O, dDense1W, dDense1B);
        dense_relu_quarter<200,500,400,600><<<dense1Grid, dense1Block>>>(
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
    print_params("CONV1_P H", hPool1O, POOL1_OUT_SIZE);
    print_params("CONV1_P D", debP1O, POOL1_OUT_SIZE);

    print_params("CONV2_W H", hConv2W, CONV2_W_SIZE);
    print_params("CONV2_W D", debC2W, CONV2_W_SIZE);
    print_params("CONV2_B H", hConv2B, CONV2_B_SIZE);
    print_params("CONV2_B D", debC2B, CONV2_B_SIZE);
    print_params("CONV2_O H", hConv2O, CONV2_OUT_SIZE);
    print_params("CONV2_O D", debC2O, CONV2_OUT_SIZE);
    print_params("CONV2_P H", hPool2O, POOL2_OUT_SIZE);
    print_params("CONV2_P D", debP2O, POOL2_OUT_SIZE);

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
        sum += gDense2O[0];
        sum += gDense2O[1];
        sum += gDense2O[2];
        sum += gDense2O[3];
        sum += gDense2O[4];
        sum += gDense2O[5];
        sum += gDense2O[6];
        sum += gDense2O[7];
        sum += gDense2O[8];
        sum += gDense2O[9];
        
        gDense2O[0] /= sum;
        gDense2O[1] /= sum;
        gDense2O[2] /= sum;
        gDense2O[3] /= sum;
        gDense2O[4] /= sum;
        gDense2O[5] /= sum;
        gDense2O[6] /= sum;
        gDense2O[7] /= sum;
        gDense2O[8] /= sum;
        gDense2O[9] /= sum;

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

        dTime += elapsedTime;


#ifdef D
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
#endif
    }

    printf("\n");
    printf("CPU: time: %f ms \n", hTime / (1+imageCount));
    printf("\n");
    
    print_all_params(hDense2O, 10);
    printf("\n");

    printf("GPU: time: %f ms \n", dTime / (1+imageCount));
    print_all_params(gDense2O, 10);
    printf("\n");

    free(hImage);

    free(hConv1W);
    free(hConv1B);
    free(hConv1O);

    free(hPool1O);

    free(hConv2W);
    free(hConv2B);
    free(hConv2O);

    free(hPool2O);

    free(hDense1W);
    free(hDense1B);
    free(hDense1O);

    free(hDense2W);
    free(hDense2B);
    free(hDense2O);

    free(gDense2O);

    deviceFree(dImage); 

    deviceFree(dConv1O); 
    deviceFree(dConv1W); 
    deviceFree(dConv1B); 

    deviceFree(dPool1O);

    deviceFree(dConv2O); 
    deviceFree(dConv2W); 
    deviceFree(dConv2B); 

    deviceFree(dPool2O);

    deviceFree(dDense1W);
    deviceFree(dDense1B);
    deviceFree(dDense1O);

    deviceFree(dDense2W);
    deviceFree(dDense2B);
    deviceFree(dDense2O);

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
