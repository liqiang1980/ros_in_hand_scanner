#include "pcl/apps/in_hand_scanner/normalestimation_withgpu.h"

#include <chrono>

void NormalEstimation_WITHGPU::get_nv_gpu(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud_ptr_xyzrgba, pcl::PointCloud <pcl::Normal>& normal) const{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_xyz (new pcl::PointCloud<pcl::PointXYZ>());
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    customized_copyPointCloud(cloud_ptr_xyzrgba,cloud_ptr_xyz);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "data change format (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin).count()) /1000000.0  <<std::endl;

    pcl::gpu::NormalEstimation::PointCloud cloud_device;
    cloud_device.upload(cloud_ptr_xyz->points);
    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    std::cout << "upload (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin).count()) /1000000.0  <<std::endl;
    pcl::gpu::NormalEstimation ne_device;
    ne_device.setInputCloud(cloud_device);
    double radius;
    int max_elements;
    radius = 0.003;
    max_elements = 5;
    ne_device.setRadiusSearch(radius, max_elements);

    pcl::gpu::NormalEstimation::Normals normals_device;
    ne_device.compute(normals_device);
    std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
    std::cout << "nv compute (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end4 - begin).count()) /1000000.0  <<std::endl;

    std::vector<PointXYZ> downloaded;
    normals_device.download(downloaded);
    std::chrono::steady_clock::time_point end5 = std::chrono::steady_clock::now();
    std::cout << "downlaod (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end5 - begin).count()) /1000000.0  <<std::endl;
    normal.resize(downloaded.size());
    for(size_t i = 0; i < downloaded.size(); ++i)
    {
        pcl::PointXYZ xyz = downloaded[i];
//        float curvature = xyz.data[3];
        normal[i].getNormalVector3fMap () = xyz.getVector3fMap ();
//        normal[i].normal_x = xyz.x;
//        normal[i].normal_y = xyz.y;
//        normal[i].normal_z = xyz.z;
    }
    std::chrono::steady_clock::time_point end6 = std::chrono::steady_clock::now();
    std::cout << "change format (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end6 - begin).count()) /1000000.0  <<std::endl;
}

void NormalEstimation_WITHGPU::customized_copyPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud_in,pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out) const{
    // Allocate enough space and copy the basics
  cloud_out->header   = cloud_in->header;
  cloud_out->width    = cloud_in->width;
  cloud_out->height   = cloud_in->height;
  cloud_out->is_dense = cloud_in->is_dense;
  cloud_out->sensor_orientation_ = cloud_in->sensor_orientation_;
  cloud_out->sensor_origin_ = cloud_in->sensor_origin_;
  cloud_out->points.resize (cloud_in->points.size ());

  if (cloud_in->points.size () == 0)
    return;

    // Iterate over each point
  for (size_t i = 0; i < cloud_in->points.size (); ++i)
    pcl::copyPoint (cloud_in->points[i], cloud_out->points[i]);
}
