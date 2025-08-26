#include "viewer.hpp"
#include <iostream>
#include <stdexcept>

namespace basics {

pcl::PointCloud<pcl::PointXYZ>::Ptr loadCloud(const std::string &path) {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    if (path.size() >= 4 && path.substr(path.size()-4) == ".pcd") {
        if (pcl::io::loadPCDFile(path, *cloud) < 0)
            throw std::runtime_error("Failed to load PCD: " + path);
    } else {
        if (pcl::io::loadPLYFile(path, *cloud) < 0)
            throw std::runtime_error("Failed to load PLY: " + path);
    }
    return cloud;
}

std::shared_ptr<pcl::visualization::PCLVisualizer>
makeViewer(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
           const std::string& title)
{
    auto vis = std::make_shared<pcl::visualization::PCLVisualizer>(title);
    vis->setBackgroundColor(0.05, 0.05, 0.08);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud, 200, 230, 255);
    vis->addPointCloud<pcl::PointXYZ>(cloud, color, "cloud");
    vis->addCoordinateSystem(0.2);
    vis->initCameraParameters();
    return vis;
}

void show(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, int delay_ms) {
    auto vis = makeViewer(cloud);
    while (!vis->wasStopped()) vis->spinOnce(delay_ms);
}

} 