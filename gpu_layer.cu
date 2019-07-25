// 各ブロックが出力の各チャネルを操作する
// 各スレッドが出力の各ピクセルの値を決定する 
template <int InSize, int InChannels, int InSize2, 
          int OutSize, int OutSize2,
          int KernelSize, int KernelSize2>
__global__ void conv2D(float* inImg, float* outImg, 
                       float* weight, float* bias)
{
    /*
        BlockDim.x == the number of output channels
        threadDim  == (inputSize.x, inputSize.y, 1)
    */
    __shared__ float sharedImg[InSize2];

    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y; 
    const unsigned int pos = tx + InSize * ty; 

    float sum = 0;

    #pragma unroll
    for (unsigned int ch = 0; ch < InChannels; ch++) {
        __syncthreads();
        sharedImg[pos] = inImg[pos + InSize2 * ch]; 
        __syncthreads();

        if (tx > OutSize+1 || tx < 2 || ty > OutSize+1 || ty < 2) {
            continue;
        }

        #pragma unroll 5
        for (unsigned int i = 0; i < 5; i++) {
            #pragma unroll 5
            for (unsigned int j = 0; j < 5; j++) {
                unsigned int kernelPos = i + KernelSize * j + KernelSize2 * ch;
                sum += sharedImg[tx-j + InSize * (ty-j)] * weight[kernelPos] + bias[kernelPos];
            }
        }
    }

    outImg[(tx-2) + (ty-2) * OutSize + blockIdx.x] = sum;
}


template<int OutSize, int OutSize2> 
__global__ void maxpool(float* inImg, float* outImg)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tx2 = tx * 2;
    const unsigned int ty2 = ty * 2;
    const unsigned int ch = blockIdx.x * OutSize2;

    outImg[tx + OutSize * ty + ch] = fmaxf(
        fmaxf( inImg[tx2 + OutSize *  ty2    + ch], inImg[(tx2+1) + OutSize *  ty2    + ch]),
        fmaxf( inImg[tx2 + OutSize * (ty2+1) + ch], inImg[(tx2+1) + OutSize * (ty2+1) + ch])
    ); 
}

template <int InSize>
__global__ void dense(float* input, float* output, float* weight, float* bias)
{
    /*
    gridDim == (output size, 1, 1)
    blockDim == (input size, 1, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;
    const unsigned int bSize = gridDim.x;

    __shared__ float sharedOut[InSize];

    sharedOut[tx] = input[tx] * weight[tx + bSize * bx]; 
    __syncthreads();

    for (unsigned int i = InSize / 2; i > 0; i >>=1){
        if (tx < i) {
            sharedOut[tx] = sharedOut[tx + i];
        }

        __syncthreads();
    }
    
    if (tx == 0){
        output[bx] = sharedOut[0] + bias[bx];
    }
}

__global__ void relu(float* input, float* output)
{
    const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    output[id] = fmaxf(0, input[id]);
}

template <int InSize>
__global__ void softmax(float* input, float* output)
{
    const unsigned int tx = threadIdx.x;
    const float exp = expf(input[tx]);

    __shared__ float sharedOut[InSize];

    sharedOut[tx] = exp;
    for (unsigned int i = InSize / 2; i > 0; i >>=1){
        if (tx < i) {
            sharedOut[tx] += sharedOut[tx + i];
        }

        __syncthreads();
    }
    
    output[tx] = exp / sharedOut[0];
}


template <int InSize>
__global__ void dense_relu(float* input, float* output, float* weight, float* bias)
{
    /*
    gridDim == (output size, 1, 1)
    blockDim == (input size, 1, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int bx = blockIdx.x;
    const unsigned int bSize = gridDim.x;

    __shared__ float sharedOut[InSize];

    sharedOut[tx] = input[tx] * weight[tx + bSize * bx]; 
    __syncthreads();

    for (unsigned int i = InSize / 2; i > 0; i >>=1){
        if (tx < i) {
            sharedOut[tx] = sharedOut[tx + i];
        }

        __syncthreads();
    }
    
    if (tx == 0){
        output[bx] = fmaxf(sharedOut[0] + bias[bx], 0);
    }
}


template <int OutChannels, int InSize2>
__global__ void dense_softmax(float* input, float* output, float* weight, float* bias)
{
    /*
    gridDim == (1, 1, 1)
    blockDim == (500, 1, 1)
    */

    const unsigned int tx = threadIdx.x;
    const unsigned int tSize = blockDim.x;
    const unsigned int tSizeh = tSize >> 1;

    const unsigned int weightPos = OutChannels * tx;
    const unsigned int channelPos = tx >= tSizeh;

    // 
    __shared__ float sharedOut[InSize2];
    __shared__ float sharedSum[OutChannels];

    #pragma unroll
    for (unsigned int ch = 0; ch < OutChannels;  ch+=2) {
        sharedOut[tx] = input[tx] * weight[weightPos + ch * tSize];
        sharedOut[tx + tSize] = input[tx] * weight[weightPos + (ch+1) * tSize];

        __syncthreads();

        unsigned int sift = 0;
        for (unsigned int i = tSize; i > 4; i >>= 1) {
            sift += channelPos ? (i >> 1) : 0;
            if (tx < i) {
                sharedOut[tx + sift] += sharedOut[tx + (i >> 1) + sift];
            }

            __syncthreads();
        }

        if (tx == 0 || tx == tSizeh) {
            sharedSum[ch + 2 * channelPos] = expf(sharedOut[tx] + bias[ch + channelPos]);
        }
    }
    
    __syncthreads();

    if (tx > OutChannels) return;

    const float exp = sharedSum[tx];
    for (unsigned int i = tSize / 2; i > 0; i >>=1){
        if (tx < i) {
            sharedSum[tx] += sharedSum[tx + i];
        }

        __syncthreads();
    }
    
    output[tx] = exp / sharedOut[0];

}
