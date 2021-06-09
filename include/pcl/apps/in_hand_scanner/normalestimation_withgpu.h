#ifndef NORMALESTIMATION_WITHGPU_H
#define NORMALESTIMATION_WITHGPU_H


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>

#include <boost/shared_ptr.hpp>

#include <pcl/cuda/features/normal_3d.h>
#include <pcl/cuda/time_cpu.h>
#include <pcl/cuda/time_gpu.h>
#include <pcl/cuda/io/cloud_to_pcl.h>
#include <pcl/cuda/io/extract_indices.h>
#include <pcl/cuda/io/disparity_to_cloud.h>
#include <pcl/cuda/io/host_device.h>


#include <iostream>

#include <pcl/gpu/features/features.hpp>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/search/search.h>

#include <pcl/common/io.h>

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace pcl::cuda;


class NormalEstimation_WITHGPU
{
  public:
    NormalEstimation_WITHGPU(){}
  public:
    void get_nv_gpu(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud_xyzrgba, pcl::PointCloud <pcl::Normal>& normal) const;
    void customized_copyPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&,pcl::PointCloud<pcl::PointXYZ>::Ptr&) const;
};


#endif // NORMALESTIMATION_WITHGPU_H
