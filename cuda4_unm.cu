// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "font24x40_lsb.h"
#define FONT_W 24
#define FONT_H 40

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y)
        return;
    if (l_x >= t_color_cuda_img.m_size.x)
        return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];

    // Store BW point to new image
    t_bw_cuda_img.m_p_uchar1[l_y * t_bw_cuda_img.m_size.x + l_x].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_grayscale<<<l_blocks, l_threads>>>(t_color_cuda_img, t_bw_cuda_img);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_insert_image(CudaImg t_cuda_big_img, CudaImg t_cuda_small_img, int2 t_pos, uchar3 mask)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_cuda_small_img.m_size.y || l_y + t_pos.y >= t_cuda_big_img.m_size.y)
        return;
    if (l_x >= t_cuda_small_img.m_size.x || l_x + t_pos.x >= t_cuda_big_img.m_size.x)
        return;

    int small_index = l_y * t_cuda_small_img.m_size.x + l_x;
    int big_index = (l_y+t_pos.y) * t_cuda_big_img.m_size.x + (l_x + t_pos.x);
    
    // Get point from color picture
    uchar4 fg = t_cuda_small_img.m_p_uchar4[small_index];
    uchar3 bg = t_cuda_big_img.m_p_uchar3[big_index];

    // l_bgr.x *= mask.x;
    // l_bgr.y *= mask.y;
    // l_bgr.z *= mask.z;

    float alpha = fg.w / 255.0f;

    uchar3 out;
    out.x = (alpha*fg.x + (1-alpha)*bg.x);
    out.y = (alpha*fg.y + (1-alpha)*bg.y);
    out.z = (alpha*fg.z + (1-alpha)*bg.z);

    // Store point at position in big image
    // t_cuda_big_img.m_p_uchar3[(t_pos.y + l_y) * t_cuda_big_img.m_size.x + (t_pos.x + l_x)] = l_bgr;
    t_cuda_big_img.at3(l_y + t_pos.y, l_x + t_pos.x, out);
}

void cu_insert_image(CudaImg &t_cuda_big_img, CudaImg &t_cuda_small_img, int2 t_pos, uchar3 mask)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((t_cuda_small_img.m_size.x + l_block_size - 1) / l_block_size, (t_cuda_small_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_insert_image<<<l_blocks, l_threads>>>(t_cuda_big_img, t_cuda_small_img, t_pos, mask);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_swap_image(CudaImg t_cuda_img1, CudaImg t_cuda_img2)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_cuda_img2.m_size.y || l_y >= t_cuda_img1.m_size.y)
        return;
    if (l_x >= t_cuda_img2.m_size.x || l_x >= t_cuda_img1.m_size.x)
        return;

    // Get point from color picture
    uchar3 l_bgr = t_cuda_img2.m_p_uchar3[l_y * t_cuda_img2.m_size.x + l_x];

    t_cuda_img2.at3(l_y, l_x, t_cuda_img1.m_p_uchar3[l_y * t_cuda_img1.m_size.x + l_x]);

    // Store point at position in big image
    t_cuda_img1.at3(l_y, l_x, l_bgr);
}

void cu_swap_image(CudaImg &t_cuda_img1, CudaImg &t_cuda_img2)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((t_cuda_img2.m_size.x + l_block_size - 1) / l_block_size, (t_cuda_img2.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_swap_image<<<l_blocks, l_threads>>>(t_cuda_img1, t_cuda_img2);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_swap2_image(CudaImg t_cuda_img1, CudaImg t_cuda_img2, int2 t_pos)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_cuda_img2.m_size.y || l_y + t_pos.y >= t_cuda_img1.m_size.y)
        return;
    if (l_x >= t_cuda_img2.m_size.x || l_x + t_pos.x >= t_cuda_img1.m_size.x)
        return;

    // Get point from color picture
    uchar3 l_bgr = t_cuda_img2.m_p_uchar3[l_y * t_cuda_img2.m_size.x + l_x];

    t_cuda_img2.at3(l_y, l_x, t_cuda_img1.m_p_uchar3[(l_y + t_pos.y) * t_cuda_img1.m_size.x + l_x + t_pos.x]);

    // Store point at position in big image
    t_cuda_img1.at3(l_y + t_pos.y, l_x + t_pos.x, l_bgr);
}

void cu_swap2_image(CudaImg &t_cuda_img1, CudaImg &t_cuda_img2, CudaImg &helper)
{
    cudaError_t l_cerr;

    int2 pos1 = make_int2(0, 0);
    int2 pos2 = make_int2(helper.m_size.x, 0);
    int2 pos3 = make_int2(0, helper.m_size.y);
    int2 pos4 = make_int2(helper.m_size.x, helper.m_size.y);

    int l_block_size = 16;
    dim3 l_blocks((t_cuda_img2.m_size.x + l_block_size - 1) / l_block_size, (t_cuda_img2.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos1);
    kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img2, helper, pos1);
    kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos1);

    // kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos2);
    // kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img2, helper, pos2);
    // kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos2);

    // kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos3);
    // kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img2, helper, pos3);
    // kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos3);

    kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos4);
    kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img2, helper, pos4);
    kernel_swap2_image<<<l_blocks, l_threads>>>(t_cuda_img1, helper, pos4);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_rotate90_image(CudaImg t_cu_img, CudaImg t_cu_img_rotated, int t_direction)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_y >= t_cu_img.m_size.y || l_x >= t_cu_img.m_size.x)
        return;

    uchar4 pixel = t_cu_img.m_p_uchar4[l_y * t_cu_img.m_size.x + l_x];

    if (t_direction == 0)
    {
        // Rotate 90° clockwise
        t_cu_img_rotated.at4(l_x, t_cu_img.m_size.y - l_y - 1, pixel);
    }
    else
    {
        // Rotate 90° counterclockwise
        t_cu_img_rotated.at4(t_cu_img.m_size.x - l_x - 1, l_y, pixel);
    }
}

void cu_rotate90(CudaImg &t_cu_img, CudaImg &t_cu_img_rotated, int t_direction)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((t_cu_img.m_size.x + l_block_size - 1) / l_block_size,
                  (t_cu_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_rotate90_image<<<l_blocks, l_threads>>>(t_cu_img, t_cu_img_rotated, t_direction);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_scale_image(CudaImg t_cu_orig, CudaImg t_cu_scaled, float2 l_scale)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_x >= t_cu_scaled.m_size.x || l_y >= t_cu_scaled.m_size.y)
        return;

    float l_orig_x = l_x * l_scale.x;
    float l_orig_y = l_y * l_scale.y;

    int x = (int)l_orig_x;
    int y = (int)l_orig_y;

    if (x >= t_cu_orig.m_size.x - 1 || y >= t_cu_orig.m_size.y - 1)
        return; // avoid overflow

    float l_diff_x = l_orig_x - x;
    float l_diff_y = l_orig_y - y;

    uchar4 bgr00 = t_cu_orig.m_p_uchar4[y * t_cu_orig.m_size.x + x];
    uchar4 bgr01 = t_cu_orig.m_p_uchar4[y * t_cu_orig.m_size.x + (1 + x)];
    uchar4 bgr10 = t_cu_orig.m_p_uchar4[(1 + y) * t_cu_orig.m_size.x + x];
    uchar4 bgr11 = t_cu_orig.m_p_uchar4[(1 + y) * t_cu_orig.m_size.x + (1 + x)];

    uchar4 bgr;

    bgr.x = bgr00.x * (1 - l_diff_y) * (1 - l_diff_x) +
            bgr01.x * (1 - l_diff_y) * (l_diff_x) +
            bgr10.x * (l_diff_y) * (1 - l_diff_x) +
            bgr11.x * (l_diff_y) * (l_diff_x);

    bgr.y = bgr00.y * (1 - l_diff_y) * (1 - l_diff_x) +
            bgr01.y * (1 - l_diff_y) * (l_diff_x) +
            bgr10.y * (l_diff_y) * (1 - l_diff_x) +
            bgr11.y * (l_diff_y) * (l_diff_x);

    bgr.z = bgr00.z * (1 - l_diff_y) * (1 - l_diff_x) +
            bgr01.z * (1 - l_diff_y) * (l_diff_x) +
            bgr10.z * (l_diff_y) * (1 - l_diff_x) +
            bgr11.z * (l_diff_y) * (l_diff_x);

    bgr.w = bgr00.w * (1 - l_diff_y) * (1 - l_diff_x) +
            bgr01.w * (1 - l_diff_y) * (l_diff_x) +
            bgr10.w * (l_diff_y) * (1 - l_diff_x) +
            bgr11.w * (l_diff_y) * (l_diff_x);

    t_cu_scaled.at4(l_y, l_x, bgr);
}

void cu_scale(CudaImg &t_cu_orig, CudaImg &t_cu_scaled)
{
    cudaError_t l_cerr;

    float2 scale = make_float2(
        (float)(t_cu_orig.m_size.x - 1) / (float)t_cu_scaled.m_size.x,
        (float)(t_cu_orig.m_size.y - 1) / (float)t_cu_scaled.m_size.y);

    int l_block_size = 16;
    dim3 l_blocks((t_cu_scaled.m_size.x + l_block_size - 1) / l_block_size,
                  (t_cu_scaled.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_scale_image<<<l_blocks, l_threads>>>(t_cu_orig, t_cu_scaled, scale);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_character_image(CudaImg t_cu_img, char t_char, int2 pos, uchar4 color)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_y + pos.y >= t_cu_img.m_size.y || l_x + pos.x >= t_cu_img.m_size.x)
        return;
    if (l_y >= FONT_H || l_x >= FONT_W)
    {
        return;
    }

    if (font[(int)t_char][l_y] & (1 << l_x))
    {
        t_cu_img.at4(l_y + pos.y, l_x + pos.x, color);
    }
}

void cu_character(CudaImg t_cu_img, char t_char, int2 pos, uchar4 color)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((FONT_W + l_block_size - 1) / l_block_size,
                  (FONT_H + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_character_image<<<l_blocks, l_threads>>>(t_cu_img, t_char, pos, color);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}