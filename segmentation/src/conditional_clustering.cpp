// Contains the ECC example from the PCL docs using the Statue_4.pcd

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <filesystem>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/search/kdtree.h>

static const std::filesystem::path dataDir = DATA_DIR;  // defined by CMake
static const std::string filename = "Statues_4.pcd";


typedef pcl::PointXYZI      PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float /*squared_distance*/)
{
  return std::fabs(point_a.intensity - point_b.intensity) < 5.0f;
}

bool
enforceNormalOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float /*squared_distance*/)
{
  Eigen::Map<const Eigen::Vector3f> na = point_a.getNormalVector3fMap();
  Eigen::Map<const Eigen::Vector3f> nb = point_b.getNormalVector3fMap();
  if (std::fabs(point_a.intensity - point_b.intensity) < 5.0f)
    return true;
  return std::fabs(na.dot(nb)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI));
}

bool
customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> na = point_a.getNormalVector3fMap();
  Eigen::Map<const Eigen::Vector3f> nb = point_b.getNormalVector3fMap();

  if (squared_distance < 10000.0f)
  {
    if (std::fabs(point_a.intensity - point_b.intensity) < 8.0f) return true;
    if (std::fabs(na.dot(nb)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI))) return true;
  }
  else
  {
    if (std::fabs(point_a.intensity - point_b.intensity) < 3.0f) return true;
  }
  return false;
}

int main()
{
  // Data containers used
  pcl::PointCloud<PointTypeIO>::Ptr   cloud_in(new pcl::PointCloud<PointTypeIO>);
  pcl::PointCloud<PointTypeIO>::Ptr   cloud_out(new pcl::PointCloud<PointTypeIO>);
  pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals(new pcl::PointCloud<PointTypeFull>);
  pcl::IndicesClustersPtr             clusters(new pcl::IndicesClusters),
                                       small_clusters(new pcl::IndicesClusters),
                                       large_clusters(new pcl::IndicesClusters);
  pcl::search::KdTree<PointTypeIO>::Ptr search_tree(new pcl::search::KdTree<PointTypeIO>);
  pcl::console::TicToc tt;

  // ---- Load via DATA_DIR / filename (like your second code) ----
  const std::filesystem::path in_path = dataDir / filename;
  const std::filesystem::path out_path = dataDir / "cloud_out";

  std::cerr << "Loading: " << in_path << "\n"; tt.tic();
  if (pcl::io::loadPCDFile(in_path.string(), *cloud_in) < 0) {
    std::cerr << "ERROR: failed to read PCD: " << in_path << "\n";
    return 1;
  }
  std::cerr << ">> Done: " << tt.toc() << " ms, " << cloud_in->size() << " points\n";

  // Downsample the cloud using a Voxel Grid class
  std::cerr << "Downsampling...\n"; tt.tic();
  pcl::VoxelGrid<PointTypeIO> vg;
  vg.setInputCloud(cloud_in);
  vg.setLeafSize(80.0f, 80.0f, 80.0f);
  vg.setDownsampleAllData(true);
  vg.filter(*cloud_out);
  std::cerr << ">> Done: " << tt.toc() << " ms, " << cloud_out->size() << " points\n";

  if (cloud_out->empty()) {
    std::cerr << "ERROR: empty cloud after voxel filter.\n";
    return 1;
  }

  // Set up a Normal Estimation class and merge data in cloud_with_normals
  std::cerr << "Computing normals...\n"; tt.tic();
  pcl::copyPointCloud(*cloud_out, *cloud_with_normals);
  pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
  ne.setInputCloud(cloud_out);
  ne.setSearchMethod(search_tree);
  ne.setRadiusSearch(300.0);
  ne.compute(*cloud_with_normals);
  std::cerr << ">> Done: " << tt.toc() << " ms\n";

  // Set up a Conditional Euclidean Clustering class
  std::cerr << "Segmenting to clusters...\n"; tt.tic();
  pcl::ConditionalEuclideanClustering<PointTypeFull> cec(true);
  cec.setInputCloud(cloud_with_normals);
  cec.setConditionFunction(&customRegionGrowing);
  cec.setClusterTolerance(500.0);
  cec.setMinClusterSize(cloud_with_normals->size() / 1000);
  cec.setMaxClusterSize(cloud_with_normals->size() / 5);
  cec.segment(*clusters);
  cec.getRemovedClusters(small_clusters, large_clusters);
  std::cerr << ">> Done: " << tt.toc() << " ms\n";

  // Using the intensity channel for lazy visualization of the output
  for (const auto& small_cluster : (*small_clusters))
    for (const auto& j : small_cluster.indices)
      (*cloud_out)[j].intensity = -2.0f;
  for (const auto& large_cluster : (*large_clusters))
    for (const auto& j : large_cluster.indices)
      (*cloud_out)[j].intensity = +10.0f;
  for (const auto& cluster : (*clusters))
  {
    int label = std::rand() % 8;
    for (const auto& j : cluster.indices)
      (*cloud_out)[j].intensity = static_cast<float>(label);
  }

  // Save the output point cloud (unchanged)
  std::cerr << "Saving...\n"; tt.tic();
  if (pcl::io::savePCDFile(out_path / "output.pcd", *cloud_out) < 0) {
    std::cerr << "ERROR: failed to save output.pcd\n";
    return 1;
  }
  std::cerr << ">> Done: " << tt.toc() << " ms\n";

  return 0;
}
