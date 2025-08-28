#include <iostream>
#include <filesystem>
#include "viewer.hpp"

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>



typedef pcl::PointXYZ PointT;

static const std::filesystem::path dataDir = DATA_DIR;
static const std::string filename = "table_scene_lms400.pcd";

int main(int, char**){
    std::cout << "Hello, from ransac!\n";

    const std::filesystem::path in_path  = dataDir / filename;
    const std::filesystem::path out_dir  = dataDir / "cloud_out";
    const std::filesystem::path out_path = out_dir / "table_scene_lms400_cylinder.pcd";
    std::filesystem::create_directories(out_dir); // ensure cloud_out exists

    // All the objects needed
    pcl::PCDReader reader;
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
    pcl::PCDWriter writer;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    pcl::VoxelGrid<PointT> vg;

    // Datasets
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_voxelised (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

    std::cout << "Reading from " <<dataDir << "/" << filename << std::endl;
    // Read data
    if (reader.read((dataDir / filename).string(), *cloud) < 0) {
        PCL_ERROR("Couldn't read file %s\n", filename.c_str());
        return -1;
    }

    std::cerr << "PointCloud has: " << cloud->points.size() << " data points." << std::endl;

    // Voxel downsample
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f); // 1 cm voxels (tune!)
    vg.filter(*cloud_voxelised);

    // Build the passthrough filter to remove noise
    pass.setInputCloud (cloud_voxelised);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-10, 10);
    pass.filter (*cloud_filtered);
    std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

    // Estimate point normals
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_filtered);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);

    // Create segmentation object for the planar model
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight (0.1);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.03);
    seg.setInputCloud (cloud_filtered);
    seg.setInputNormals (cloud_normals);

    // Obtain plane inliers and coefficients
    seg.segment (*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from input cloud
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers_plane);
    extract.setNegative (true);
    extract.filter (*cloud_filtered2);
    extract_normals.setInputCloud (cloud_normals);
    extract_normals.setIndices (inliers_plane);
    extract_normals.setNegative (true);
    extract_normals.filter (*cloud_normals2);

    // Create segmentation object for cylinder
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_CYLINDER);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight (0.1);
    seg.setMaxIterations (10000);
    seg.setDistanceThreshold (0.05);
    seg.setRadiusLimits (0, 0.1);
    seg.setInputCloud (cloud_filtered2);
    seg.setInputNormals (cloud_normals2);

    // Obtain cylinder inliers and coefficients
    seg.segment (*inliers_cylinder, *coefficients_cylinder);
    std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

    // Write extracted cylinder to disk
    extract.setInputCloud (cloud_filtered2);
    extract.setIndices (inliers_cylinder);
    extract.setNegative (false);
    pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT>());
    extract.filter (*cloud_cylinder);

    if (cloud_cylinder->points.empty())
        std::cerr << "Can't find the cylindrical component." << std::endl;
    else
    {
        Eigen::Vector4f c;
        pcl::compute3DCentroid(*cloud_cylinder, c);
        std::cout << "C0: " << c[0] << " ,C1: " << c[1] << " ,C2: " << c[2] <<std::endl;
        

        std::cerr << "PointCloud representing the cylindrical component: " 
                  << cloud_cylinder->points.size () << " data points." << std::endl;
        writer.write (out_path.string(), *cloud_cylinder, false);

        auto vis = basics::makeViewer(cloud_filtered2, "RANSAC — scene");

        // highlight cylinder points
        pcl::visualization::PointCloudColorHandlerCustom<PointT> cyl_color(cloud_cylinder, 20, 200, 20);
        vis->addPointCloud<PointT>(cloud_cylinder, cyl_color, "cylinder");
        vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cylinder");

        // centroid marker (assuming you computed Eigen::Vector4f c earlier)
        vis->addSphere(pcl::PointXYZ(c[0], c[1], c[2]), 0.02, 1.0, 0.1, 0.1, "centroid_sphere");
        vis->addText3D("centroid", pcl::PointXYZ(c[0], c[1], c[2]), 0.05, 1.0, 0.1, 0.1, "centroid_label");

        // run the viewer loop
        while (!vis->wasStopped()) vis->spinOnce(16);
    }

    return 0;
}
