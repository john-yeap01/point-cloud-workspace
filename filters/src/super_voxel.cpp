#include <iostream>
#include <filesystem>
#include <sstream>

#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

// VTK for drawing graph lines
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

void addSupervoxelConnectionsToViewer(PointT &supervoxel_center,
                                      PointCloudT &adjacent_supervoxel_centers,
                                      std::string supervoxel_name,
                                      pcl::visualization::PCLVisualizer::Ptr &viewer);

// ---- Fixed input path ----
// Define at compile time, e.g. -DDATA_DIR=\"/abs/path\"
static const std::filesystem::path dataDir = DATA_DIR;   // must be a quoted macro
static const std::string filename = "sectionA.pcd";

int main() {
  const std::filesystem::path in_path   = dataDir / filename;
  const std::filesystem::path out_dir   = dataDir / "cloud_out";
  std::filesystem::create_directories(out_dir);

  pcl::console::print_highlight("Loading point cloud from fixed path:\n  %s\n",
                                in_path.string().c_str());

  // Try to load as XYZRGBA first; if that fails, load XYZ and convert.
  PointCloudT::Ptr cloud(new PointCloudT);
  {
    pcl::PCLPointCloud2 blob;
    pcl::PCDReader reader;
    if (reader.read(in_path.string(), blob) < 0) {
      pcl::console::print_error("Failed to read %s\n", in_path.string().c_str());
      return 1;
    }

    // Check fields to decide conversion
    std::string fields = pcl::getFieldsList(blob);
    bool has_rgba = (fields.find("rgba") != std::string::npos) ||
                    (fields.find("rgb")  != std::string::npos);

    if (has_rgba) {
      pcl::fromPCLPointCloud2(blob, *cloud);
    } else {
      // Convert XYZ -> XYZRGBA with a constant color
      pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromPCLPointCloud2(blob, *xyz);
      cloud->reserve(xyz->size());
      for (const auto &p : xyz->points) {
        PointT q;
        q.x = p.x; q.y = p.y; q.z = p.z;
        // white opaque
        q.r = 255; q.g = 255; q.b = 255; q.a = 255;
        cloud->push_back(q);
      }
      cloud->width  = static_cast<uint32_t>(cloud->size());
      cloud->height = 1u;
      cloud->is_dense = xyz->is_dense;
    }
    pcl::console::print_info("Loaded %u points (fields: %s)\n",
                             static_cast<unsigned>(cloud->size()), fields.c_str());
  }

  // ---- Supervoxel parameters (fixed, but tweak here if you want) ----
  const float voxel_resolution  = 0.6f;  // 20 cm base voxels (good starting point for UAV LiDAR)
  const float seed_resolution   = 0.9f;  // seeds ~40 cm apart
  const float color_importance  = 0.0f;   // LiDAR typically no color; keep 0
  const float spatial_importance= 1.0f;
  const float normal_importance = 1.0f;

  pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
  // If your cloud is not from a single camera, keep this false to avoid weird warps
  super.setUseSingleCameraTransform(false);
  super.setInputCloud(cloud);
  super.setColorImportance(color_importance);
  super.setSpatialImportance(spatial_importance);
  super.setNormalImportance(normal_importance);

  std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
  pcl::console::print_highlight("Extracting supervoxels...\n");
  super.extract(supervoxel_clusters);
  pcl::console::print_info("Found %zu supervoxels\n", supervoxel_clusters.size());

  // ---- Visualization ----
  auto viewer = pcl::visualization::PCLVisualizer::Ptr(
      new pcl::visualization::PCLVisualizer("Supervoxels"));
  viewer->setBackgroundColor(0, 0, 0);

  PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud();
  viewer->addPointCloud(voxel_centroid_cloud, "voxel_centroids");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                           2.0, "voxel_centroids");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                           0.95, "voxel_centroids");

  PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud();
  viewer->addPointCloud(labeled_voxel_cloud, "labeled_voxels");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                           0.8, "labeled_voxels");

  // Optional normals:
  // PointNCloudT::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud(supervoxel_clusters);
  // viewer->addPointCloudNormals<PointNT>(sv_normal_cloud, 1, 0.05f, "sv_normals");

  pcl::console::print_highlight("Building supervoxel adjacency graph...\n");
  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  super.getSupervoxelAdjacency(supervoxel_adjacency);

  for (auto label_itr = supervoxel_adjacency.cbegin();
       label_itr != supervoxel_adjacency.cend(); ) {
    uint32_t label = label_itr->first;
    auto sv = supervoxel_clusters.at(label);

    PointCloudT neighbors;
    auto range = supervoxel_adjacency.equal_range(label);
    for (auto it = range.first; it != range.second; ++it) {
      auto nb = supervoxel_clusters.at(it->second);
      neighbors.push_back(nb->centroid_);
    }

    std::stringstream ss; ss << "sv_edges_" << label;
    addSupervoxelConnectionsToViewer(sv->centroid_, neighbors, ss.str(), viewer);

    label_itr = supervoxel_adjacency.upper_bound(label);
  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(50);
  }
  return 0;
}

void addSupervoxelConnectionsToViewer(PointT &center,
                                      PointCloudT &neighbors,
                                      std::string name,
                                      pcl::visualization::PCLVisualizer::Ptr &viewer) {
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

  for (auto &q : neighbors) {
    points->InsertNextPoint(center.data);
    points->InsertNextPoint(q.data);
  }

  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  polyData->SetPoints(points);
  polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints());
  for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
    polyLine->GetPointIds()->SetId(i, i);
  cells->InsertNextCell(polyLine);
  polyData->SetLines(cells);
  viewer->addModelFromPolyData(polyData, name);
}
