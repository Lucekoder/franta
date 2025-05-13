// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include <opencv2/core/mat.hpp>

// Structure definition for exchanging data between Host and Device
struct CudaImg
{
  uint3 m_size; // size of picture
  union
  {
    void *m_p_void;     // data of picture
    uchar1 *m_p_uchar1; // data of picture
    uchar3 *m_p_uchar3; // data of picture
    uchar4 *m_p_uchar4; // data of picture
  };
  CudaImg() : m_p_void(nullptr)
  {
    m_size = make_uint3(0, 0, 0);
  }
  CudaImg(uint3 m_size, void *data_ptr) : m_p_void(data_ptr), m_size(m_size) {}
  CudaImg(int height, int width, void *data_ptr) : m_p_void(data_ptr)
  {
    this->m_size.y = height;
    this->m_size.x = width;
  }

  __device__ void at3(int y, int x, uchar3 value)
  {
    if (y >= 0 && y < this->m_size.y && x >= 0 && x < this->m_size.x)
    {
      this->m_p_uchar3[y * this->m_size.x + x] = value;
    }
    else
    {
      printf("Error: Index (%d, %d) is out of bounds.\n", y, x);
    }
  }
  __device__ void at4(int y, int x, uchar4 value)
  {
    if (y >= 0 && y < this->m_size.y && x >= 0 && x < this->m_size.x)
    {
      this->m_p_uchar4[y * this->m_size.x + x] = value;
    }
    else
    {
      printf("Error: Index (%d, %d) is out of bounds.\n", y, x);
    }
  }
  __device__ void at1(int y, int x, uchar1 value)
  {
    if (y >= 0 && y < this->m_size.y && x >= 0 && x < this->m_size.x)
    {
      this->m_p_uchar1[y * this->m_size.x + x] = value;
    }
    else
    {
      printf("Error: Index (%d, %d) is out of bounds.\n", y, x);
    }
  }

  cv::Mat toCvMat1() const
  {
    cv::Mat img(m_size.y, m_size.x, CV_8UC1);
    cudaMemcpy(img.data, m_p_uchar1, m_size.x * m_size.y * sizeof(uchar1), cudaMemcpyDeviceToHost);
    return img;
  }

  // Converts uchar3 image to cv::Mat (BGR)
  cv::Mat toCvMat3() const
  {
    cv::Mat img(m_size.y, m_size.x, CV_8UC3);
    cudaMemcpy(img.data, m_p_uchar3, m_size.x * m_size.y * sizeof(uchar3), cudaMemcpyDeviceToHost);
    return img;
  }

  // Converts uchar4 image to cv::Mat (BGRA)
  cv::Mat toCvMat4() const
  {
    cv::Mat img(m_size.y, m_size.x, CV_8UC4);
    cudaMemcpy(img.data, m_p_uchar4, m_size.x * m_size.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
    return img;
  }
};
