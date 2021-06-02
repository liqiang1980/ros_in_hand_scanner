#include "pcl/apps/in_hand_scanner/normalestimation_withgpu.h"


void NormalEstimation_WITHGPU::get_nv_gpu(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud_ptr_xyzrgba){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_xyz;
    pcl::copyPointCloud(*cloud_ptr_xyzrgba,*cloud_ptr_xyz);
    pcl::gpu::NormalEstimation::PointCloud cloud_device;
    cloud_device.upload(cloud_ptr_xyz->points);

    pcl::gpu::NormalEstimation ne_device;
    ne_device.setInputCloud(cloud_device);
    double radius;
    int max_elements;
    radius = 0.1;
    max_elements = 3000;
    ne_device.setRadiusSearch(radius, max_elements);

    pcl::gpu::NormalEstimation::Normals normals_device;
    ne_device.compute(normals_device);

    std::vector<PointXYZ> downloaded;
    normals_device.download(downloaded);

    for(size_t i = 0; i < downloaded.size(); ++i)
    {

        pcl::PointXYZ xyz = downloaded[i];
        float curvature = xyz.data[3];

//        float abs_error = 0.01f;
//        ASSERT_NEAR(n.normal_x, xyz.x, abs_error);
//        ASSERT_NEAR(n.normal_y, xyz.y, abs_error);
//        ASSERT_NEAR(n.normal_z, xyz.z, abs_error);

//        float abs_error_curv = 0.01f;
//        ASSERT_NEAR(n.curvature, curvature, abs_error_curv);
    }
}
