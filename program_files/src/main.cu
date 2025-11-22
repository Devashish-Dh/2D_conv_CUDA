#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <getopt.h>

#include <cstring> 
#include "timers.h"

#include "../include/conv_kernels.cuh"


#include "separability.hpp"
// for variant 4:
#include <cmath>




// Forward declarations
void cpu_conv2d_same(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size);
float max_abs_diff(const float* array_a, const float* array_b, size_t num_elements);

// Utility functions
bool load_images_bin(const std::string& path, std::vector<float>& output, 
                     int& batch_size, int& height, int& width);
void gen_random_images(std::vector<float>& output, int batch_size, int height, 
                       int width, unsigned seed = 42);

static void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::puts("  --n=N             batch size (default 8)\n"
              "  --h=H             image height (default 1024)\n"
              "  --w=W             image width (default 1024)\n"
              "  --k=K             kernel size - must be odd (default 5)\n"
              "  --impl=NAME       implementation: baseline|variant1|variant2|variant3|variant4|variant5|bonus\n"
              "  --iters=I         number of timing iterations (default 5)\n"
              "  --verify          compare GPU results with CPU reference\n"
              "  --images=PATH     load images from binary file (otherwise generate random)\n"
              "  --batch=B         batch size for streams (default 16)\n"
              "  --streams=S       number of CUDA streams (default 2)\n");
}





#define MAX_KERNEL_SIZE 16




//for variant 4:
__constant__ float const_kernel_row[MAX_KERNEL_SIZE];
__constant__ float const_kernel_col[MAX_KERNEL_SIZE];










__constant__ float const_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// have to do duplication of some funcs due to constant memory not working with linker (extern) and rdc=true disabled;

static __device__ __forceinline__ size_t idx3_(
                                                int batch_index, 
                                                int row, 
                                                int col,
                                                int height, 
                                                int width) 
{
    return static_cast<size_t>(batch_index) * height * width + static_cast<size_t>(row) * width + col; // converts the 3D coords into the flat array coords. 
    //(acc to the batch, the row, the col of the input 3d coords)
}


__global__ void kernel_conv2d_variant2(
    const float* __restrict__ input_images,
    float* __restrict__ output_images,
    int batch_size,
    int height,
    int width,
    int kernel_size)
{
    if (kernel_size <= 0 || kernel_size > MAX_KERNEL_SIZE) return;

    const int blockW = blockDim.x;   // e.g. 32
    const int blockH = blockDim.y;   // e.g. 8

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // top-left anchor for kernel (to handle even and odd sized kernels) (not required but did it for future proofing)

    const int anchor = (kernel_size - 1) / 2; // floor((k-1)/2)
    // tile width/height must cover entire kernel span on both sides:
    const int tileW = blockW + (kernel_size - 1); // blockW + k - 1
    const int tileH = blockH + (kernel_size - 1); // blockH + k - 1

    extern __shared__ float shared_tile[]; // size: tileW * tileH floats

    // origin of the output tile (global coords)
    const int out_row_start = by * blockH;
    const int out_col_start = bx * blockW;

    //load of the input tile (with zero-padding) into the shared memory
    for (int y = ty; y < tileH; y += blockH) 
    {       
        //less divergence 
        for (int x = tx; x < tileW; x += blockW) 
        {
            int in_row = min(max(out_row_start + y - anchor, 0), height - 1);
            int in_col = min(max(out_col_start + x - anchor, 0), width - 1);

            float v = __ldg(&input_images[idx3_(bz, in_row, in_col, height, width)]);

            // Zero out halo regions explicitly
            if ((out_row_start + y - anchor) < 0 || 
                (out_row_start + y - anchor) >= height ||
                (out_col_start + x - anchor) < 0 ||
                (out_col_start + x - anchor) >= width) {
                v = 0.0f;
            }

            shared_tile[y * tileW + x] = v;
        }

        
        // has some divergence that can be eliminated
        // int in_row = out_row_start + y - anchor; // global input row
        // for (int x = tx; x < tileW; x += blockW) 
        // {
        //     int in_col = out_col_start + x - anchor; // global input col
        //     float v = 0.0f;
        //     if ((bz < batch_size) && (in_row >= 0) && (in_row < height) && (in_col >= 0) && (in_col < width)) 
        //     {
        //         v = __ldg(&input_images[idx3_(bz, in_row, in_col, height, width)]);
        //     }
            
        //     shared_tile[y * tileW + x] = v;
        // }

    }
    __syncthreads();

    // global coords of this thread's output pixel
    const int out_r = out_row_start + ty;
    const int out_c = out_col_start + tx;
    if (bz >= batch_size || out_r >= height || out_c >= width) return;

    // local center position (in shared tile) uses the anchor
    const int local_r = ty + anchor;
    const int local_c = tx + anchor;

    float acc = 0.0f;

    // convolution: tile_row simplifies to ty + kr
    for (int kr = 0; kr < kernel_size; ++kr) 
    {
        const int kernel_row_offset = kr * kernel_size;
        const int tile_row = local_r + (kr - anchor); // equals ty + kr
        
        #pragma unroll  // may help for small kernels
        for (int kc = 0; kc < kernel_size; ++kc) 
        {
            const int tile_col = local_c + (kc - anchor); // equals tx + kc
            float in_val = shared_tile[tile_row * tileW + tile_col];
            float w = const_kernel[kernel_row_offset + kc];
            acc = fmaf(in_val, w, acc);
        }
    }

    output_images[idx3_(bz, out_r, out_c, height, width)] = acc;
}



void conv2d_variant2(const float* input, 
                     float* output,
                     int batch_size, 
                     int height, 
                     int width, 
                     int kernel_size, // MUST use this
                     cudaStream_t stream) 
{
    
    //do i need to adjust constant?
    if (kernel_size <= 0 || kernel_size > MAX_KERNEL_SIZE) return;

    dim3 threads_per_block(32, 8, 1);
    
    const int blockW = threads_per_block.x; // 32
    const int blockH = threads_per_block.y; // 8
    const int radius = kernel_size / 2;     // The radius depends on the input kernel_size

    // These must match the tile size calculation inside the kernel
    const int tileW = blockW + 2 * radius;  
    const int tileH = blockH + 2 * radius;  

    // Size required in BYTES
    size_t shared_mem_size = (size_t)tileH * tileW * sizeof(float);

    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    kernel_conv2d_variant2<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(
        input, output, batch_size, height, width, kernel_size
    );
}


#define Reuse_F 4

__global__ void kernel_conv2d_variant3(
                                        const float* __restrict__ input_images,
                                        float* __restrict__ output_images,
                                        int batch_size, 
                                        int height, 
                                        int width, 
                                        int kernel_size)
{
    if (kernel_size <= 0 || kernel_size > MAX_KERNEL_SIZE) return;

    extern __shared__ float s_input[]; // shared tile: shared_h x shared_w (row-major)

    const int radius = (kernel_size - 1) / 2;

    const int tile_out_w = blockDim.x * Reuse_F;
    const int tile_out_h = blockDim.y;

    const int block_origin_col = blockIdx.x * tile_out_w;
    const int block_origin_row = blockIdx.y * tile_out_h;
    const int batch_index = blockIdx.z;

    if ((unsigned)batch_index >= (unsigned)batch_size) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_col = block_origin_col + tx * Reuse_F;
    const int out_row = block_origin_row + ty;

    const int shared_w = tile_out_w + 2 * radius;
    const int shared_h = tile_out_h + 2 * radius;

    const int global_row_base = block_origin_row - radius;
    const int global_col_base = block_origin_col - radius;

    // Cooperative load of the shared tile (zero-padding for OOB)
    for (int trow = ty; trow < shared_h; trow += blockDim.y) 
    {
        int global_row = global_row_base + trow;

        for (int tcol = tx; tcol < shared_w; tcol += blockDim.x) {
            int global_col = global_col_base + tcol;

            float v = 0.0f;
            if ((unsigned)global_row < (unsigned)height &&
                (unsigned)global_col < (unsigned)width) 
            {
                size_t gidx = idx3_(batch_index, global_row, global_col, height, width);
                v = __ldg(&input_images[gidx]);
            }

            s_input[trow * shared_w + tcol] = v;
        }
    }
    __syncthreads();

    if ((unsigned)out_row >= (unsigned)height) return; // row outside image -> nothing to do

    // accumulators for contiguous outputs
    float acc[Reuse_F];
    #pragma unroll
    for (int i = 0; i < Reuse_F; ++i) acc[i] = 0.0f;

    // leftmost shared-column index corresponding to the leftmost output produced by this thread within shared tile coordinates (no +radius because shared includes halo)
    const int local_left = tx * Reuse_F;

    // convolution loop (optimized)
    for (int kr = 0; kr < kernel_size; ++kr) {

        int srow = ty + kr;
        // Hoist the shared row pointer (1 multiply per kr instead of per FMA)
        float* srow_ptr = &s_input[srow * shared_w];

        int krow_offset = kr * kernel_size;

        for (int kc = 0; kc < kernel_size; ++kc) {

            float kw = const_kernel[krow_offset + kc];

            // pointer to the leftmost pixel for this thread’s outputs
            float* ptr = srow_ptr + (local_left + kc);

            #pragma unroll
            for (int i = 0; i < Reuse_F; ++i) {
                // pointer walk — no index arithmetic in inner loop
                float in_val = ptr[i];
                acc[i] = fmaf(in_val, kw, acc[i]);
            }
        }
    }


    // write outputs (with checking horizontal bounds)
    #pragma unroll
    for (int i = 0; i < Reuse_F; ++i) {
        int f_out_col = out_col + i;
        if ((unsigned)f_out_col < (unsigned)width) {
            size_t out_idx = idx3_(batch_index, out_row, f_out_col, height, width);
            output_images[out_idx] = acc[i];
        }
    }
}


void conv2d_variant3(const float* input, 
                     float* output,
                     int batch_size, 
                     int height, 
                     int width, 
                     int kernel_size,
                     cudaStream_t stream)
{
    if (kernel_size > MAX_KERNEL_SIZE || (kernel_size % 2) == 0) {
        fprintf(stderr, "Error: kernel_size %d invalid or exceeds MAX_KERNEL_SIZE %d\n",
                kernel_size, MAX_KERNEL_SIZE);
        return;
    }

    // block / tile configuration
    dim3 threads_per_block(32, 8, 1);

    const int tile_out_w = threads_per_block.x * Reuse_F;
    const int tile_out_h = threads_per_block.y;

    dim3 blocks_per_grid(
        (width  + tile_out_w - 1) / tile_out_w,
        (height + tile_out_h - 1) / tile_out_h,
        batch_size
    );

    const int radius = (kernel_size - 1) / 2;
    const int shared_w = tile_out_w + 2 * radius;
    const int shared_h = tile_out_h + 2 * radius;

    size_t shared_bytes = (size_t)shared_w * shared_h * sizeof(float);

    kernel_conv2d_variant3<<<blocks_per_grid, threads_per_block, shared_bytes, stream>>>(
        input, output, batch_size, height, width, kernel_size
    );
}


































































































//Variant4 :
#define REUSE_F 4

__global__ void kernel_conv2d_variant4_row(
    const float* __restrict__ input,
    float* __restrict__ intermediate,
    int batch_size,
    int height,
    int width,
    int kernel_size)
{
    extern __shared__ float s_input[];
    const int radius = (kernel_size - 1) / 2;

    const int tile_out_w = blockDim.x * REUSE_F;
    const int tile_out_h = blockDim.y;

    const int block_col = blockIdx.x * tile_out_w;
    const int block_row = blockIdx.y * tile_out_h;
    const int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int shared_w = tile_out_w + 2 * radius;
    const int shared_h = tile_out_h;

    // Cooperative load into shared memory (including halo)
    for (int row = ty; row < shared_h; row += blockDim.y)
    {
        int global_row = block_row + row;
        for (int col = tx; col < shared_w; col += blockDim.x)
        {
            int global_col = block_col - radius + col;
            float val = 0.0f;
            if (global_row >= 0 && global_row < height &&
                global_col >= 0 && global_col < width)
            {
                val = __ldg(&input[idx3_(batch_idx, global_row, global_col, height, width)]);
            }
            s_input[row * shared_w + col] = val;
        }
    }
    __syncthreads();

    const int out_row = block_row + ty;
    const int out_col = block_col + tx * REUSE_F;

    float acc[REUSE_F] = {0};

    // Horizontal convolution
    for (int k = 0; k < kernel_size; ++k)
    {
        float kw = const_kernel_row[k];
        #pragma unroll
        for (int i = 0; i < REUSE_F; ++i)
        {
            int shared_col = tx * REUSE_F + i + k;
            acc[i] = fmaf(s_input[ty * shared_w + shared_col], kw, acc[i]);
        }
    }

    // Write results to intermediate buffer
    #pragma unroll
    for (int i = 0; i < REUSE_F; ++i)
    {
        int col_idx = out_col + i;
        if (out_row < height && col_idx < width)
        {
            intermediate[idx3_(batch_idx, out_row, col_idx, height, width)] = acc[i];
        }
    }
}

__global__ void kernel_conv2d_variant4_col(
    const float* __restrict__ intermediate,
    float* __restrict__ output,
    int batch_size,
    int height,
    int width,
    int kernel_size)
{
    extern __shared__ float s_input[];
    const int radius = (kernel_size - 1) / 2;

    const int tile_out_w = blockDim.x;
    const int tile_out_h = blockDim.y * REUSE_F;

    const int block_col = blockIdx.x * tile_out_w;
    const int block_row = blockIdx.y * tile_out_h;
    const int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int shared_w = tile_out_w;
    const int shared_h = tile_out_h + 2 * radius;

    // Load vertical halo into shared memory
    for (int row = ty; row < shared_h; row += blockDim.y)
    {
        int global_row = block_row - radius + row;
        for (int col = tx; col < shared_w; col += blockDim.x)
        {
            int global_col = block_col + col;
            float val = 0.0f;
            if (global_row >= 0 && global_row < height &&
                global_col >= 0 && global_col < width)
            {
                val = __ldg(&intermediate[idx3_(batch_idx, global_row, global_col, height, width)]);
            }
            s_input[row * shared_w + col] = val;
        }
    }
    __syncthreads();

    const int out_row = block_row + ty * REUSE_F;
    const int out_col = block_col + tx;

    float acc[REUSE_F] = {0};

    // Vertical convolution
    for (int k = 0; k < kernel_size; ++k)
    {
        float kw = const_kernel_col[k];
        #pragma unroll
        for (int i = 0; i < REUSE_F; ++i)
        {
            int shared_row = ty * REUSE_F + i + k;
            acc[i] = fmaf(s_input[shared_row * shared_w + tx], kw, acc[i]);
        }
    }

    // Write results to output buffer
    #pragma unroll
    for (int i = 0; i < REUSE_F; ++i)
    {
        int row_idx = out_row + i;
        if (row_idx < height && out_col < width)
        {
            output[idx3_(batch_idx, row_idx, out_col, height, width)] = acc[i];
        }
    }
}




void conv2d_variant4(
    const float* input,
    float* /*unused*/ device_kernel,
    float* device_intermediate,
    float* output,
    int batch_size,
    int height,
    int width,
    int kernel_size,
    cudaStream_t stream)
{
    dim3 threads_row(64, 4, 1);
    dim3 blocks_row((width + threads_row.x*REUSE_F - 1)/(threads_row.x*REUSE_F),
                    (height + threads_row.y - 1)/threads_row.y,
                    batch_size);

    size_t shared_bytes_row = (threads_row.x*REUSE_F + kernel_size - 1) * threads_row.y * sizeof(float);

    kernel_conv2d_variant4_row<<<blocks_row, threads_row, shared_bytes_row, stream>>>(
        input, device_intermediate, batch_size, height, width, kernel_size
    );

    dim3 threads_col(16, 16, 1);
    dim3 blocks_col((width + threads_col.x - 1)/threads_col.x,
                    (height + threads_col.y*REUSE_F - 1)/(threads_col.y*REUSE_F),
                    batch_size);

    size_t shared_bytes_col = threads_col.x * (threads_col.y*REUSE_F + kernel_size - 1) * sizeof(float);

    kernel_conv2d_variant4_col<<<blocks_col, threads_col, shared_bytes_col, stream>>>(
        device_intermediate, output, batch_size, height, width, kernel_size
    );
}






//________________________________________________________________________________________________________________________________________________________________________________________________________






// safe free helpers
auto safeFreeDevice = [](void*& p) {
    if (p) {
        cudaError_t e = cudaFree(p);
        if (e != cudaSuccess) fprintf(stderr, "cudaFree error: %s\n", cudaGetErrorString(e));
        p = nullptr;
    }
};
auto safeFreeHost = [](void*& p) {
    if (p) {
        cudaError_t e = cudaFreeHost(p);
        if (e != cudaSuccess) fprintf(stderr, "cudaFreeHost error: %s\n", cudaGetErrorString(e));
        p = nullptr;
    }
};


int main(int argc, char** argv) {

    //GPU available : NVIDIA RTX A6000    



    // Default parameters
    int batch_size = 8;
    int height = 1024;
    int width = 1024;
    int kernel_size = 5;
    int num_iterations = 5;
    bool run_verification = false;
    std::string implementation = "naive";
    std::string image_path = "";

    // Command line option definitions
    const option long_options[] = {
        {"n",       required_argument, nullptr, 'n'},
        {"h",       required_argument, nullptr, 'h'},
        {"w",       required_argument, nullptr, 'w'},
        {"k",       required_argument, nullptr, 'k'},
        {"impl",    required_argument, nullptr, 'i'},
        {"iters",   required_argument, nullptr, 't'},
        {"verify",  no_argument,       nullptr, 'v'},
        {"images",  required_argument, nullptr, 'f'},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    while (true) {
        int option_char = getopt_long(argc, argv, "", long_options, nullptr);
        if (option_char == -1) {
            break;
        }
        
        switch (option_char) {
            case 'n': batch_size = std::atoi(optarg); break;
            case 'h': height = std::atoi(optarg); break;
            case 'w': width = std::atoi(optarg); break;
            case 'k': kernel_size = std::atoi(optarg); break;
            case 'i': implementation = optarg; break;
            case 't': num_iterations = std::atoi(optarg); break;
            case 'v': run_verification = true; break;
            case 'f': image_path = optarg; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Validate kernel size
    if (kernel_size % 2 == 0 || kernel_size <= 0) {
        std::fprintf(stderr, "Error: kernel size must be odd and positive\n");
        return 1;
    }

    // ===== Load or generate input images =====
    std::vector<float> host_images;
    int loaded_batch_size = batch_size;
    int loaded_height = height;
    int loaded_width = width;
    
    if (!image_path.empty() && load_images_bin(image_path, host_images, 
                                                loaded_batch_size, loaded_height, loaded_width)) {
        batch_size = loaded_batch_size;
        height = loaded_height;
        width = loaded_width;
        std::printf("Loaded images: batch=%d height=%d width=%d\n", 
                    batch_size, height, width);
    } else {
        gen_random_images(host_images, batch_size, height, width, 1234);
        std::printf("Generated random images: batch=%d height=%d width=%d\n", 
                    batch_size, height, width);
    }


    // Calculating dims : 
        
    size_t image_bytes = static_cast<size_t>(batch_size) * height * width * sizeof(float);
    size_t output_bytes = image_bytes;
    size_t kernel_bytes = static_cast<size_t>(kernel_size) * kernel_size * sizeof(float);
    




    //### Allocating pinned memory on the HOST:
    float* pinned_host_images = nullptr;
    float* pinned_host_kernel = nullptr;
    CK(cudaMallocHost(&pinned_host_images, image_bytes));  // pinned input
    CK(cudaMallocHost(&pinned_host_kernel, kernel_bytes)); // pinned kernel

    // ===== Initialize kernel (convolution filter) =====
    std::vector<float> host_kernel(kernel_size * kernel_size);
    float kernel_value = 1.0f / (kernel_size * kernel_size);
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        host_kernel[i] = kernel_value;
    }



    // Copy generated/loaded vector into pinned buffer
    std::memcpy(pinned_host_images, host_images.data(), image_bytes);
    std::memcpy(pinned_host_kernel, host_kernel.data(), kernel_bytes);



    // ===== Allocate host output buffers =====
    std::vector<float> host_output(batch_size * height * width);
    std::vector<float> host_reference_output;

    // ===== Allocate device memory =====
    float* device_input = nullptr;
    float* device_kernel = nullptr;
    float* device_output = nullptr;
    //variant4
    float* device_intermediate = nullptr;



    CK(cudaMalloc(&device_input, image_bytes));
    CK(cudaMalloc(&device_output, output_bytes));
    
    // USING THE PINNED MEMORY HERE-------------------------------------------------------:
    // Copy input data to device
    CK(cudaMemcpy(device_input, pinned_host_images, image_bytes, cudaMemcpyHostToDevice));

    //contingent on the variant USING THEM:

    // CK(cudaMemcpy(device_kernel, pinned_host_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    // //copy into the constant memory
    // CK(cudaMemcpyToSymbol(const_kernel, pinned_host_kernel, kernel_bytes, 0, cudaMemcpyHostToDevice));

    //------------------------------------------------------------------------------------:


    // // Copy input data to device
    // CK(cudaMemcpy(device_input, host_images.data(), image_bytes, cudaMemcpyHostToDevice));
    // CK(cudaMemcpy(device_kernel, host_kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));

    
    // //copy into the constant memory
    // CK(cudaMemcpyToSymbol(const_kernel, host_kernel.data(), kernel_bytes, 0, cudaMemcpyHostToDevice));

    //cudaDeviceSynchronize();



    // ===== Generate CPU reference if verification requested =====
    if (run_verification) {
        std::printf("Computing CPU reference (this may take a while)...\n");
        host_reference_output.resize(static_cast<size_t>(batch_size) * height * width);
        cpu_conv2d_same(host_images.data(), host_kernel.data(), host_reference_output.data(), 
                        batch_size, height, width, kernel_size);
    }


    // !!!! only copy over data IF USING THAT (saving some mem bandwidth)
    if (implementation == "naive"    ||
        implementation == "baseline" ||
        implementation == "variant1")
        {
            CK(cudaMalloc(&device_kernel, kernel_bytes));
            CK(cudaMemcpy(device_kernel, pinned_host_kernel, kernel_bytes, cudaMemcpyHostToDevice));
        }

    if (implementation == "variant2" ||
        implementation == "variant3" )
        {
            CK(cudaMemcpyToSymbol(const_kernel, pinned_host_kernel,
                                kernel_bytes, 0, cudaMemcpyHostToDevice));

                // //for debugging the constant memory: 
                // int count = 0;
                // std::vector<float> debug_kernel(MAX_KERNEL_SIZE * MAX_KERNEL_SIZE);
                
                // CK(cudaMemcpyFromSymbol(debug_kernel.data(), const_kernel, kernel_bytes, 0, cudaMemcpyDeviceToHost));
                
                // for (int i = 0; i < kernel_size * kernel_size; ++i)
                // {   count++; 
                //     //printf("kernel(flattened): %f ", debug_kernel[i]);
                // }

                // printf("\n");
                // printf("ConstantMem:_kernel_vals_count: %d",count);
                // printf("\n");
                        


        }

    if (implementation == "variant4")
        {
            if (kernel_size > MAX_KERNEL_SIZE) 
            {
                std::fprintf(stderr, "Error: separable kernel_size > MAX_KERNEL_SIZE\n");
                exit(1);
            }

            std::vector<float> row_filter, col_filter;
            bool sep = is_separable(host_kernel, kernel_size, row_filter, col_filter);
            
            if (sep)
            {
            std::cout << "Kernel is separable : Variant4 applicable, switching to 2-pass conv!" << std::endl;


            CK(cudaMemcpyToSymbol(const_kernel_row, row_filter.data(),
                                kernel_size * sizeof(float) , 0, cudaMemcpyHostToDevice));
            
            CK(cudaMemcpyToSymbol(const_kernel_col, col_filter.data(),
                                kernel_size * sizeof(float) , 0, cudaMemcpyHostToDevice));
            
            
                                
            // // Debug constant memory for row filter
            // std::vector<float> debug_row(MAX_KERNEL_SIZE);
            // CK(cudaMemcpyFromSymbol(debug_row.data(), const_kernel_row, 
            //                         kernel_size * sizeof(float), 0, cudaMemcpyDeviceToHost));

            // printf("Constant row kernel values (size=%d):\n", kernel_size);
            // for (int i = 0; i < kernel_size; ++i) {
            //     printf("%f ", debug_row[i]);
            // }
            // printf("\n");

            // // Debug constant memory for column filter
            // std::vector<float> debug_col(MAX_KERNEL_SIZE);
            // CK(cudaMemcpyFromSymbol(debug_col.data(), const_kernel_col, 
            //                         kernel_size * sizeof(float), 0, cudaMemcpyDeviceToHost));

            // printf("Constant col kernel values (size=%d):\n", kernel_size);
            // for (int i = 0; i < kernel_size; ++i) {
            //     printf("%f ", debug_col[i]);
            // }
            // printf("\n");




            CK(cudaMalloc(&device_intermediate, image_bytes));

            
            }

            else
        {
            std::cout << "Kernel NOT separable Variant4 NOT applicable." << std::endl;
            
            safeFreeDevice(reinterpret_cast<void*&>(device_intermediate));
            safeFreeHost(reinterpret_cast<void*&>(pinned_host_images));
            safeFreeHost(reinterpret_cast<void*&>(pinned_host_kernel));
            safeFreeDevice(reinterpret_cast<void*&>(device_input));
            safeFreeDevice(reinterpret_cast<void*&>(device_kernel));
            safeFreeDevice(reinterpret_cast<void*&>(device_output));
                        
            std::printf("\n");
            return 0;

            }
        }




    // ===== Helper function to select and call the right kernel =====
    auto call_kernel = [&](const std::string& impl_name, cudaStream_t stream = 0) {
        if (impl_name == "naive" || impl_name == "baseline") {
            conv2d_baseline(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant1") {
            conv2d_variant1(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant2") {
            conv2d_variant2(device_input, device_output, 
                            batch_size, height, width, kernel_size, stream);    // for this and all going forward, device_kernel is redundant
        } else if (impl_name == "variant3") {
            conv2d_variant3(device_input, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant4") {
            conv2d_variant4(device_input, device_kernel,device_intermediate, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else {
            std::fprintf(stderr, "Unknown implementation: %s\n", impl_name.c_str());
            std::exit(2);
        }
    };

    // ===== Run benchmark with proper timing =====
    std::printf("\n========== BENCHMARK: %s ==========\n", implementation.c_str());
    std::printf("Configuration: N=%d, H=%d, W=%d, k=%d, iters=%d\n", 
                batch_size, height, width, kernel_size, num_iterations);
    std::printf("Total pixels per iteration: %.2f MPix\n", 
                static_cast<double>(batch_size) * height * width / 1e6);
    
    // Warmup runs (2-3 iterations to initialize GPU)
    std::printf("Running warmup...\n");
    for (int i = 0; i < 3; ++i) {
        call_kernel(implementation);
    }

    CK(cudaDeviceSynchronize());
    
    // Timed benchmark runs
    std::printf("Running timed iterations...\n");
    CudaEventTimer timer;
    std::vector<float> iteration_times;
    
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        timer.record_start();
        call_kernel(implementation);
        float elapsed_ms = timer.record_stop_and_elapsed_ms();
        iteration_times.push_back(elapsed_ms);
        
        std::printf("  Iteration %d: %.4f ms\n", iteration + 1, elapsed_ms);
    }
    
    // Calculate statistics
    float total_time = 0.0f;
    float min_time = iteration_times[0];
    float max_time = iteration_times[0];
    
    for (float time : iteration_times) {
        total_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    float average_time = total_time / num_iterations;
    double megapixels = static_cast<double>(batch_size) * height * width / 1e6;
    double throughput = megapixels / (average_time / 1000.0);
    
    // Print results
    std::printf("\n========== RESULTS ==========\n");
    std::printf("IMPLEMENTATION: %s\n", implementation.c_str());
    std::printf("Average time:   %.4f ms\n", average_time);
    std::printf("Min time:       %.4f ms\n", min_time);
    std::printf("Max time:       %.4f ms\n", max_time);
    std::printf("Throughput:     %.2f MPix/s\n", throughput);
    std::printf("Ops per pixel:  %d (kernel size %dx%d)\n", 
                kernel_size * kernel_size, kernel_size, kernel_size);

    // Verify correctness against CPU reference
    if (run_verification) {
        std::printf("\n========== VERIFICATION ==========\n");
        CK(cudaMemcpy(host_output.data(), device_output, output_bytes, 
                      cudaMemcpyDeviceToHost));
        
        float max_difference = max_abs_diff(host_output.data(), 
                                            host_reference_output.data(), 
                                            static_cast<size_t>(batch_size) * height * width);
        
        std::printf("Maximum absolute difference vs CPU: %.8f\n", max_difference);
        
        if (max_difference < 1e-4) {
            std::printf("✓ PASS: Results match CPU reference\n");
        } else {
            std::printf("✗ FAIL: Results differ from CPU reference\n");
        }
    }




    safeFreeHost(reinterpret_cast<void*&>(pinned_host_images));
    safeFreeHost(reinterpret_cast<void*&>(pinned_host_kernel));
    safeFreeDevice(reinterpret_cast<void*&>(device_intermediate));
    safeFreeDevice(reinterpret_cast<void*&>(device_input));
    safeFreeDevice(reinterpret_cast<void*&>(device_kernel));
    safeFreeDevice(reinterpret_cast<void*&>(device_output));

    // //variant4 cleanup
    // cudaFree(device_intermediate);

    // //More Cleanup
    // cudaFreeHost(pinned_host_images);
    // cudaFreeHost(pinned_host_kernel);


    // // Cleanup
    // cudaFree(device_input);
    // cudaFree(device_kernel);
    // cudaFree(device_output);
    
    std::printf("\n");
    return 0;
}