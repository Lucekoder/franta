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
// Image manipulation is performed by OpenCV library.
//
// ***********************************************************************

#include <stdio.h>
#include <random>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv
{
}

// Function prototype from .cu file
void cu_run_grayscale(CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img);
void cu_insert_image(CudaImg &t_cuda_big_img, CudaImg &t_cuda_small_img, int2 t_pos, uchar3 mask);
void cu_swap_image(CudaImg &t_cuda_img1, CudaImg &t_cuda_img2);
void cu_swap2_image(CudaImg &t_cuda_img1, CudaImg &t_cuda_img2, CudaImg &helper);
void cu_rotate90(CudaImg &t_cu_img, CudaImg &t_cu_img_rotated, int t_direction);
void cu_scale(CudaImg &t_cu_orig, CudaImg &t_cu_scaled);
void cu_character(CudaImg t_cu_img, char t_char, int2 pos, uchar4 color);

int main(int t_numarg, char **t_arg)
{
    // Generate random number
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(100, 400);

    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (t_numarg < 3)
    {
        printf("Enter 2 picture filenames!\n");
        return 1;
    }

    // Load image
    cv::Mat trakar = cv::imread(t_arg[1], cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_COLOR );
    if (!trakar.data)
    {
        printf("Unable to read file '%s'\n", t_arg[1]);
        return 1;
    }
    cv::Mat img2 = cv::imread(t_arg[2], cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_COLOR );
    if (!img2.data)
    {
        printf("Unable to read file '%s'\n", t_arg[2]);
        return 1;
    }

    // data for CUDA
    CudaImg cuda_pampeliska(trakar.size().height, trakar.size().width, (uchar4 *)trakar.data);
    CudaImg cuda_img2(img2.size().height, img2.size().width, (uchar3 *)img2.data);

    // generate random position for dundelion to be inserted at

    // cv::imshow("default", cuda_img2.toCvMat3());
    do
    {
        //  Scale Trakar Start
        printf("Original Width:\t%d\nOriginal Height:\t%d\n", cuda_pampeliska.m_size.x, cuda_pampeliska.m_size.y);
        int newWidth = distr(gen);
        int newHeight = ((float)cuda_pampeliska.m_size.y / (float)cuda_pampeliska.m_size.x) * (float)newWidth;
        printf("Width:\t%d\nHeight:\t%d\n", newWidth, newHeight);

        cv::Mat scaled_data(cv::Size(newHeight, newWidth), CV_8UC4);
        cv::Mat rotated_data(cv::Size(cuda_pampeliska.m_size.y, cuda_pampeliska.m_size.x), CV_8UC4);

        CudaImg scaledImg(scaled_data.size().height, scaled_data.size().width, scaled_data.data);

        CudaImg rotatedImg(rotated_data.size().height, rotated_data.size().width, rotated_data.data);

        // Scale Trakar End

        std::string str_width = std::to_string(newWidth);

        // insert width
        // cu_character(scaledImg, 'A', {(newWidth-24)/2, 20}, {255, 0, 0, 255});
        for (int i = 0; i < str_width.length(); i++)
        {
            int2 pos = make_int2((cuda_pampeliska.m_size.x - 24 * str_width.length()) / 2 + i * 24, 0);
            cu_character(cuda_pampeliska, str_width[i], pos, {255, 0, 0, 255});
        }

        // cv::imshow("char1", cuda_pampeliska.toCvMat4());
        cu_rotate90(cuda_pampeliska, rotatedImg, 1);

        // rotate and insert height
        str_width = std::to_string(newHeight);

        for (int i = 0; i < str_width.length(); i++)
        {
            int2 pos = make_int2((rotatedImg.m_size.x - 24 * str_width.length()) / 2 + i * 24, 0);
            cu_character(rotatedImg, str_width[i], pos, {255, 0, 0, 255});
        }

        // cv::imshow("char2", rotatedImg.toCvMat4());
        cu_scale(rotatedImg, scaledImg);
        // rotate back

        // cu_rotate90(rotatedImg, scaledImg, 1);
        std::uniform_int_distribution<> distr_pos_x(0, cuda_img2.m_size.x - scaledImg.m_size.y);
        std::uniform_int_distribution<> distr_pos_y(0, cuda_img2.m_size.y - scaledImg.m_size.x);

        int2 insert_pos = make_int2(distr_pos_x(gen), distr_pos_y(gen));

        cu_insert_image(cuda_img2, scaledImg, insert_pos, {1, 1, 1});
        cv::imshow("inserted", cuda_img2.toCvMat3());

    } while (cv::waitKey(100) != 'q');

    // Show the Color and BW image
    // cv::imshow( "Color", l_bw_cv_img );
}

// Practice vypisy

// cv::Mat data( large_img.size()/2, CV_8U );

//     CudaImg helper(data.size().height, data.size().width, ( uchar3 * ) data.data);

// // Function calling from .cu file
//     // // cu_run_grayscale( l_bgr_cuda_img, l_bw_cuda_img );
//     // cu_insert_image( l_large_cuda_img, l_small_cuda_img, {100, 100}, {1,1,1});
//     // // cv::imshow( "normal", large_img );
//     // cu_insert_image( l_large_cuda_img, l_small_cuda_img, {100+(int)l_small_cuda_img.m_size.x, 100}, {0,0,1});
//     // // cv::imshow( "red", large_img );
//     // cu_insert_image( l_large_cuda_img, l_small_cuda_img, {100, 100+(int)l_small_cuda_img.m_size.y}, {0,1,0});
//     // // cv::imshow( "green", large_img );
//     // cu_insert_image( l_large_cuda_img, l_small_cuda_img, {100+(int)l_small_cuda_img.m_size.x, 100+(int)l_small_cuda_img.m_size.y}, {1,0,0});
//     // // cv::imshow( "blue", large_img );

//     // cv::imshow( "insert all", large_img );
//     // cu_swap2_image(l_large_cuda_img, l_small_cuda_img, helper);
//     // cv::imshow( "After", large_img );

//     cv::Mat r_data( cv::Size(large_img.size().height, large_img.size().width), CV_8UC3 );

//     CudaImg rotated(r_data.size().height, r_data.size().width, ( uchar3* ) r_data.data);

//     cv::Mat scaled_data( cv::Size(large_img.size().width*2, large_img.size().height), CV_8UC3);

//     CudaImg scaled(scaled_data.size().height, scaled_data.size().width, ( uchar3* ) scaled_data.data);

//     // cv::imshow("before", l_large_cuda_img.toCvMat3());
//     // cu_rotate90(l_large_cuda_img, rotated, 1);
//     // cv::imshow("after", rotated.toCvMat3());

//     // cu_scale(l_large_cuda_img, scaled);
//     // cv::imshow("scaled", scaled.toCvMat3());

//     cu_character(l_large_cuda_img, 'A', {100, 100}, {255, 255, 255});
//     cv::imshow("A", l_large_cuda_img.toCvMat3());
