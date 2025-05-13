#pragma once
#include "cuda_img.h"
#include <vector_types.h>

class Animation
{
public:
  CudaImg m_bg_cuda_img, m_ins_cuda_img, m_rot_cuda_img, m_res_cuda_img;
  int m_initialized;

  Animation() : m_initialized(0) {}

  void start(CudaImg &t_bg_pic, CudaImg &t_ins_pic, CudaImg &t_rot_cuda_img);

  /// @brief Generates next animation frame
  /// @param t_res_pic result picture
  /// @param t_ins_cuda_img original insert picture
  /// @param angle angle around which the insert picture will be rotated
  /// @param t_position position in which the picture will be inserted
  /// @param t_mask filter for colors of inserted picture
  void next(CudaImg &t_res_pic, float angle, int2 t_position, uchar3 t_mask);

  void stop();
};
