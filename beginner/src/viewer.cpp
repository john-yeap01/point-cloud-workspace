#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

static pcl::PointCloud<pcl::PointXYZ>::Ptr loadCloud(const std::string &path) {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    if (path.size() >= 4 && path.substr(path.size()-4) == ".pcd") {
        if (pcl::io::loadPCDFile(path, *cloud) < 0) throw std::runtime_error("Failed to load PCD: " + path);
    } else {
        if (pcl::io::loadPLYFile(path, *cloud) < 0) throw std::runtime_error("Failed to load PLY: " + path);
    }
    return cloud;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <cloud.pcd|cloud.ply>\n"; return 1; }

    auto cloud = loadCloud(argv[1]);
    std::cout << "Loaded " << cloud->size() << " points\n";

    pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer("Viewer"));
    vis->setBackgroundColor(0.05, 0.05, 0.08);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud, 200, 230, 255);
    vis->addPointCloud<pcl::PointXYZ>(cloud, color, "cloud");
    vis->addCoordinateSystem(0.2);
    vis->initCameraParameters();

    while (!vis->wasStopped()) vis->spinOnce(10);

    return 0;
}
