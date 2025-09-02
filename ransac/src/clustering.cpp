#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>   // <-- added
#include <iomanip>   // for setw, setfill
#include <filesystem>
#include <random>    // for colors
#include <chrono>

int main ()
{
    auto start = std::chrono::high_resolution_clock::now();
  // I/O paths (uses CMake-injected DATA_DIR like your other files)
  static const std::filesystem::path dataDir = DATA_DIR;
  static const std::string flag = "REMOVE PLANE";
  static const std::string filename = "forest4_segmented.pcd";
  const std::filesystem::path in_path  = dataDir / filename;
  const std::filesystem::path out_dir  = dataDir / "cloud_out";
  const std::filesystem::path clu_dir  = out_dir / "clusters";
  std::filesystem::create_directories(clu_dir);

  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  if (reader.read (in_path.string(), *cloud) < 0) {
    PCL_ERROR("Couldn't read file %s\n", in_path.string().c_str());
    return -1;
  }
  std::cout << "PointCloud before filtering has: " << cloud->size () << " data points." << std::endl;

  // Downsample (1 cm)
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.01f, 0.01f, 0.01f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->size ()  << " data points." << std::endl;

  // Plane removal (largest planes until 30% left)
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int nr_points = (int) cloud_filtered->size ();
  while (cloud_filtered->size () > 0.3 * nr_points && flag=="REMOVE PLANE"){
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.empty())
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);

    // keep plane → cloud_plane
    extract.setNegative (false);
    extract.filter (*cloud_plane);
    std::cout << "Plane component: " << cloud_plane->size () << " pts." << std::endl;

    // remove plane → cloud_f
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  // input cloud: cloud_filtered
    std::vector<int> indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_nan(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_no_nan, indices);

std::cout << "Removed " << (cloud_filtered->size() - cloud_no_nan->size())
          << " NaN points\n";

  // Kd-tree + Euclidean clustering
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.2); // 2cm
  ec.setMinClusterSize (20);
//   ec.setMinClusterSize (20);
//   ec.setMaxClusterSize (25000);
  ec.setMaxClusterSize (20000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  // Store clusters for visualization
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  clusters.reserve(cluster_indices.size());

  int j = 0;
  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_cluster->reserve(cluster.indices.size());
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    }
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "Cluster " << j << ": " << cloud_cluster->size () << " pts." << std::endl;

    // save to DATA_DIR/cloud_out/clusters/cloud_cluster_XXXX.pcd
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << j;
    const std::filesystem::path out_path = clu_dir / ("cloud_cluster_" + ss.str() + ".pcd");
    writer.write<pcl::PointXYZ> (out_path.string(), *cloud_cluster, false);

    clusters.push_back(cloud_cluster);
    ++j;
  }

  // === Colored visualization of ALL clusters ===
  if (!clusters.empty()) {
    pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer("Clusters"));
    vis->setBackgroundColor(0,0,0); // black

    // Optional: show the remaining non-plane cloud in grey beneath
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> base_color(cloud_filtered, 140, 140, 140);
    vis->addPointCloud<pcl::PointXYZ>(cloud_filtered, base_color, "base");
    vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "base");

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> U(40, 235);

    for (size_t i=0; i<clusters.size(); ++i) {
      int r = U(rng), g = U(rng), b = U(rng);
      auto& C = clusters[i];
      std::string id = "cluster_" + std::to_string(i);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clr(C, r, g, b);
      vis->addPointCloud<pcl::PointXYZ>(C, clr, id);
      vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, id);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    while (!vis->wasStopped()) vis->spinOnce(16);
  } else {
    std::cout << "No clusters found to visualize.\n";
  }

  return 0;
}
