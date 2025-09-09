// region_growing_with_fs.cpp
// Region Growing Segmentation (PCL tutorial) + your file read/write system.
// Algorithm unchanged; only I/O adapted (DATA_DIR paths, PCDReader/Writer, save outputs).
// RED POINTS IN THE VIZ MEANS THEY WERE IGNORED BECAUSE TOO MANY/TOO FEW
#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>                 // PCDReader/PCDWriter
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h>    // pcl::removeNaNFromPointCloud (indices variant)
#include <pcl/segmentation/region_growing.h>

static const std::filesystem::path dataDir = DATA_DIR;   // must be a quoted macro
static const std::string filename = "sectionA.pcd";

int main()
{
  // --- File system: set up paths + reader/writer
  const std::filesystem::path in_path  = dataDir / filename;
  const std::filesystem::path out_dir  = dataDir / "cloud_out";
  std::filesystem::create_directories(out_dir);

  pcl::PCDReader reader;
  pcl::PCDWriter writer;

  // --- Original tutorial code (logic untouched), just use reader.read
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if (reader.read(in_path.string(), *cloud) != 0)
  {
    std::cout << "Cloud reading failed: " << in_path << std::endl;
    return (-1);
  }

  std::cout << "Cloud size : " << cloud->width * cloud -> height << std::endl;

  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (60);
  normal_estimator.compute (*normals);

  // Keep indices-based NaN handling exactly like the tutorial
  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::removeNaNFromPointCloud(*cloud, *indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;

  reg.setMinClusterSize (2000);
  reg.setMaxClusterSize (100000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (60);
  reg.setInputCloud (cloud);
  reg.setIndices (indices);
  reg.setInputNormals (normals);

  // VERY IMPORTANT PARAMETERS
  reg.setSmoothnessThreshold (5.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (0.1);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  if (!clusters.empty())
  {
    std::cout << "First cluster has " << clusters[0].indices.size () << " points." << std::endl;
    std::cout << "These are the indices of the points of the initial" <<
      std::endl << "cloud that belong to the first cluster:" << std::endl;
    std::size_t counter = 0;
    while (counter < clusters[0].indices.size ())
    {
      std::cout << clusters[0].indices[counter] << ", ";
      counter++;
      if (counter % 10 == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  // Colored output (same as tutorial) + save to disk per your I/O system
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  const auto colored_path = out_dir / "region_growing_colored.pcd";
  writer.write<pcl::PointXYZRGB>(colored_path.string(), *colored_cloud, false);
  std::cout << "Saved colored segmentation to: " << colored_path << std::endl;

  // (Optional) Save first cluster’s points as a PCD for convenience — does not change algorithm
  if (!clusters.empty())
  {
    pcl::PointCloud<pcl::PointXYZ> first_cluster_pts;
    first_cluster_pts.reserve(clusters[0].indices.size());
    for (int idx : clusters[0].indices) first_cluster_pts.push_back((*cloud)[idx]);
    first_cluster_pts.width = first_cluster_pts.size(); first_cluster_pts.height = 1; first_cluster_pts.is_dense = true;

    const auto first_cluster_path = out_dir / "region_growing_first_cluster.pcd";
    writer.write<pcl::PointXYZ>(first_cluster_path.string(), first_cluster_pts, false);
    std::cout << "Saved first cluster points to: " << first_cluster_path << std::endl;
  }

  // with this:
    #ifdef __APPLE__
    // Use PCLVisualizer on the main thread to avoid Cocoa NSWindow crash
    pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer("Cluster viewer"));
    vis->setBackgroundColor(0, 0, 0);
    vis->addPointCloud<pcl::PointXYZRGB>(colored_cloud, "colored");
    vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "colored");
    vis->addCoordinateSystem(1.0);
    vis->initCameraParameters();
    while (!vis->wasStopped()) {
        vis->spinOnce(16);
    }
    #else
    pcl::visualization::CloudViewer viewer ("Cluster viewer");
    viewer.showCloud(colored_cloud);
    while (!viewer.wasStopped ()) {}
    #endif

  return (0);
}
