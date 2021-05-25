#ifndef NORMALESTIMATION_WITHGPU_H
#define NORMALESTIMATION_WITHGPU_H

// #include "opencv2/opencv.hpp"
// #include "opencv2/gpu/gpu.hpp"

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

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>

#include <pcl/gpu/features/features.hpp>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/search/search.h>

#include "pcl/apps/in_hand_scanner/input_data_processing.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace pcl::cuda;


class NormalEstimation_WITHGPU
{
  public:
    NormalEstimation_WITHGPU(): viewer ("PCL CUDA - NormalEstimation"), new_cloud(false), go_on(true) {}
  public:
    void viz_cb (pcl::visualization::PCLVisualizer& viz)
        {
          static bool first_time = true;
          double psize = 1.0,opacity = 1.0,linesize =1.0;
          std::string cloud_name ("cloud");
          boost::mutex::scoped_lock l(m_mutex);
          if (new_cloud)
          {
            //typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> ColorHandler;
            typedef pcl::visualization::PointCloudColorHandlerGenericField <pcl::PointXYZRGBNormal> ColorHandler;
            //ColorHandler Color_handler (normal_cloud);
            ColorHandler Color_handler (normal_cloud,"curvature");
            if (!first_time)
            {
              viz.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, linesize, cloud_name);
              viz.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cloud_name);
              viz.getPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, psize, cloud_name);
              //viz.removePointCloud ("normalcloud");
              viz.removePointCloud ("cloud");
            }
            else
              first_time = false;

            //viz.addPointCloudNormals<pcl::PointXYZRGBNormal> (normal_cloud, 139, 0.1, "normalcloud");
            viz.addPointCloud<pcl::PointXYZRGBNormal> (normal_cloud, Color_handler, std::string("cloud"), 0);
            viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, linesize, cloud_name);
            viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cloud_name);
            viz.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, psize, cloud_name);
            new_cloud = false;
          }
        }

        template <template <typename> class Storage> void
        file_cloud_cb (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud)
        {
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr output (new pcl::PointCloud<pcl::PointXYZRGB>);
          PointCloudAOS<Host> data_host;
          data_host.points.resize (cloud->points.size());
          for (size_t i = 0; i < cloud->points.size (); ++i)
          {
            PointXYZRGB pt;
            pt.x = cloud->points[i].x;
            pt.y = cloud->points[i].y;
            pt.z = cloud->points[i].z;
            // Pack RGB into a float
            pt.rgb = *(float*)(&cloud->points[i].rgb);
            data_host.points[i] = pt;
          }
          data_host.width = cloud->width;
          data_host.height = cloud->height;
          data_host.is_dense = cloud->is_dense;
          typename PointCloudAOS<Storage>::Ptr data = toStorage<Host, Storage> (data_host);

          // we got a cloud in device..

          boost::shared_ptr<typename Storage<float4>::type> normals;
          float focallength = 580/2.0;
          {
            ScopeTimeCPU time ("Normal Estimation");
            normals = computePointNormals<Storage, typename PointIterator<Storage,PointXYZRGB>::type > (data->points.begin (), data->points.end (), focallength, data, 0.05, 30);
          }
          go_on = false;

          boost::mutex::scoped_lock l(m_mutex);
          normal_cloud.reset (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          toPCL (*data, *normals, *normal_cloud);
          new_cloud = true;
        }

        template <template <typename> class Storage> void
        cloud_cb (const boost::shared_ptr<openni_wrapper::Image>& image,
                  const boost::shared_ptr<openni_wrapper::DepthImage>& depth_image,
                  float constant)
        {
          static int smoothing_nr_iterations = 10;
          static int smoothing_filter_size = 2;
          cv::namedWindow("Parameters", CV_WINDOW_NORMAL);
          cvCreateTrackbar ("iterations", "Parameters", &smoothing_nr_iterations, 50, NULL);
          cvCreateTrackbar ("filter_size", "Parameters", &smoothing_filter_size, 10, NULL);
          cv::waitKey (2);

          static unsigned count = 0;
          static double last = getTime ();
          double now = getTime ();
          if (++count == 30 || (now - last) > 5)
          {
            std::cout << "Average framerate: " << double(count)/double(now - last) << " Hz..........................................." <<  std::endl;
            count = 0;
            last = now;
          }

          pcl::PointCloud<pcl::PointXYZRGB>::Ptr output (new pcl::PointCloud<pcl::PointXYZRGB>);
          typename PointCloudAOS<Storage>::Ptr data;

          ScopeTimeCPU timer ("All: ");
          // Compute the PointCloud on the device
          d2c.compute<Storage> (depth_image, image, constant, data, false, 1, smoothing_nr_iterations, smoothing_filter_size);
          //d2c.compute<Storage> (depth_image, image, constant, data, true, 2);

          boost::shared_ptr<typename Storage<float4>::type> normals;
          float focallength = 580/2.0;
          {
            ScopeTimeCPU time ("Normal Estimation");
            normals = computeFastPointNormals<Storage> (data);
            //normals = computePointNormals<Storage, typename PointIterator<Storage,PointXYZRGB>::type > (data->points.begin (), data->points.end (), focallength, data, 0.05, 30);
          }

          boost::mutex::scoped_lock l(m_mutex);
          normal_cloud.reset (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
          toPCL (*data, *normals, *normal_cloud);
          new_cloud = true;
        }

        void
        run (bool use_device, bool use_file)
        {
          if (use_file)
          {
            pcl::Grabber* filegrabber = 0;

            float frames_per_second = 1;
            bool repeat = false;

            std::string path = "./frame_0.pcd";
            filegrabber = new pcl::PCDGrabber<pcl::PointXYZRGB > (path, frames_per_second, repeat);

            if (use_device)
            {
              std::cerr << "[NormalEstimation] Using GPU..." << std::endl;
              boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f = boost::bind (&NormalEstimation::file_cloud_cb<Device>, this, _1);
              filegrabber->registerCallback (f);
            }
            else
            {
              std::cerr << "[NormalEstimation] Using CPU..." << std::endl;
              boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f = boost::bind (&NormalEstimation::file_cloud_cb<Host>, this, _1);
              filegrabber->registerCallback (f);
            }

            filegrabber->start ();
            while (go_on)//!viewer.wasStopped () && go_on)
            {
              pcl_sleep (1);
            }
            filegrabber->stop ();
          }
          else
          {
            pcl::Grabber* grabber = new pcl::OpenNIGrabber();

            boost::signals2::connection c;
            if (use_device)
            {
              std::cerr << "[NormalEstimation] Using GPU..." << std::endl;
              boost::function<void (const boost::shared_ptr<openni_wrapper::Image>& image, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_image, float)> f = boost::bind (&NormalEstimation::cloud_cb<Device>, this, _1, _2, _3);
              c = grabber->registerCallback (f);
            }
            else
            {
              std::cerr << "[NormalEstimation] Using CPU..." << std::endl;
              boost::function<void (const boost::shared_ptr<openni_wrapper::Image>& image, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_image, float)> f = boost::bind (&NormalEstimation::cloud_cb<Host>, this, _1, _2, _3);
              c = grabber->registerCallback (f);
            }

            viewer.runOnVisualizationThread (boost::bind(&NormalEstimation::viz_cb, this, _1), "viz_cb");

            grabber->start ();

            while (!viewer.wasStopped ())
            {
              pcl_sleep (1);
            }

            grabber->stop ();
          }
        }
        void get_nv_gpu(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_xyzrgba);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normal_cloud;
        DisparityToCloud d2c;
        pcl::visualization::CloudViewer viewer;
        boost::mutex m_mutex;
        bool new_cloud, go_on;
};


#endif // NORMALESTIMATION_WITHGPU_H
