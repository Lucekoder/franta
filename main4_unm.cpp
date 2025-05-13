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
#include <sys/time.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <random>
#include <string>

#include "cuda_img.h"
#include "animation.h"

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
void cu_run_rotate(CudaImg &t_cv_img_orig, CudaImg &t_cv_img_rotate, float t_angle);

void prepare_rotated_image(cv::Mat &orig, cv::Mat &rotated, float angle)
{
    float theta = angle; // in radians
    float abs_cos = fabs(cosf(theta));
    float abs_sin = fabs(sinf(theta));

    int orig_w = orig.cols;
    int orig_h = orig.rows;

    int new_w = (int)(orig_w * abs_cos + orig_h * abs_sin);
    int new_h = (int)(orig_w * abs_sin + orig_h * abs_cos);

    rotated = cv::Mat::zeros(new_h, new_w, orig.type());
}

int main(int t_numarg, char **t_arg)
{
    Animation l_animation;
    // Generate random number
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<> distr(100, 400);

    if (t_numarg < 3)
    {
        printf("Enter 2 picture filenames!\n");
        printf("1st image - bg image\n");
        printf("2nd image - insert image\n");
        return 1;
    }

    // Load bg image
    cv::Mat l_bg_cv_img = cv::imread(t_arg[1], cv::IMREAD_UNCHANGED);
    if (!l_bg_cv_img.data)
    {
        printf("Unable to read file '%s'\n", t_arg[1]);
        return 1;
    }
    // Load insert image
    cv::Mat l_insert_cv_img = cv::imread(t_arg[2], cv::IMREAD_UNCHANGED);
    if (!l_insert_cv_img.data)
    {
        printf("Unable to read file '%s'\n", t_arg[2]);
        return 1;
    }

    printf("'%s' channels %d\n", t_arg[2], l_insert_cv_img.channels());

    if (l_insert_cv_img.channels() != 4) {    
        cv::cvtColor(l_insert_cv_img, l_insert_cv_img, cv::COLOR_BGR2BGRA);
    }

    
    cv::Mat l_rotated_cv_img = cv::Mat::zeros(l_insert_cv_img.size().height * M_SQRT2, l_insert_cv_img.size().width * M_SQRT2, l_insert_cv_img.type());

    // Create Cuda bg image
    CudaImg l_bg_cuda_img(l_bg_cv_img.size().height, l_bg_cv_img.size().width, (uchar3 *)l_bg_cv_img.data);

    // Create Cuda insert image
    CudaImg l_insert_cuda_img(l_insert_cv_img.size().height, l_insert_cv_img.size().width, (uchar4 *)l_insert_cv_img.data);


    CudaImg l_rotated_cuda_img(l_rotated_cv_img.size().height, l_rotated_cv_img.size().width, (uchar4 *)l_rotated_cv_img.data);

    // Prepare data for animation
    l_animation.start(l_bg_cuda_img, l_insert_cuda_img, l_rotated_cuda_img);

    // Simulation of rolling inserted image
    const float speed = 150.0f; // pixels per second
    const float radius = std::min(l_insert_cuda_img.m_size.x, l_insert_cuda_img.m_size.y) / 2.0f;

    const float rotation_speed = (speed / radius); // degrees per second

    float pos_x = 0.0f; // Current position of inserted image on bg image
    float pos_y = l_bg_cuda_img.m_size.y - l_rotated_cv_img.rows;
    float angle = 0.0f; // Current rotation of inserted image

    // Animation data
    int l_run_simulation = 1;
    int l_iterations = 0;
    // int l_runs = 0;

    timeval l_start_time, l_cur_time, l_old_time, l_delta_time;
    gettimeofday(&l_old_time, NULL);
    l_start_time = l_old_time;

    while (l_run_simulation)
    {
        cv::waitKey( 1 );
        
        // Update image
        int2 pos = make_int2(pos_x, pos_y); // Position of insert image

        l_animation.next(l_bg_cuda_img, angle, pos, {255, 255, 255});


        // time measuring
        gettimeofday(&l_cur_time, NULL);
        timersub(&l_cur_time, &l_old_time, &l_delta_time);
        float l_delta_sec = l_delta_time.tv_sec + l_delta_time.tv_usec / 1E6f; // time in seconds
        if (l_delta_sec < 0.001f)
            continue; // too short time
        l_old_time = l_cur_time;

        l_iterations++;

        // Update data
        pos_x += speed * l_delta_sec;
        angle -= rotation_speed * l_delta_sec;
        if (pos_x > l_bg_cuda_img.m_size.x)
        {
            l_run_simulation = 0; // end the simulation
        }

        cv::imshow("Rolling Ball", l_bg_cv_img);
        printf("pos_x = %.2f, angle = %.2f, delta_sec = %.6f\n", pos_x, angle, l_delta_sec);

    }

    l_animation.stop();

    gettimeofday( &l_cur_time, NULL );
    timersub( &l_cur_time, &l_start_time, &l_delta_time );
    int l_delta_ms = l_delta_time.tv_sec * 1000 + l_delta_time.tv_usec / 1000;

    printf( "Ball stopped after %d iterations\n", l_iterations );
    printf( "The whole simulation time %d ms.\n", l_delta_ms );
    // printf( "The ball moved through the screen %d times.\n", l_runs );
    
    cv::waitKey( 0 );
}