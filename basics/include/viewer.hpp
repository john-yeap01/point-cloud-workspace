#pragma once
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


namespace basics {

pcl::PointCloud<pcl::PointXYZ>::Ptr loadCloud(const std::string& path);

// Create a viewer with a cloud added (does not spin the loop)
std::shared_ptr<pcl::visualization::PCLVisualizer>
makeViewer(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
           const std::string& title = "Viewer");

// Convenience that runs the spin loop (returns when window is closed)
void show(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, int delay_ms = 10);

} // namespace basics