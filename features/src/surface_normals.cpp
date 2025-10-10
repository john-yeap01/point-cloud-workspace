#include <iostream>
#include <filesystem>
#include <string>
#include <thread>
#include <chrono>
#include <pcl/console/time.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/concatenate.h>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>              // fromPCLPointCloud2
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

#ifndef DATA_DIR
#  error "Define DATA_DIR via CMake: target_compile_definitions(your_target PRIVATE DATA_DIR=\"/abs/path/to/data\")"
#endif

static const std::filesystem::path kDataDir  = DATA_DIR;            // quoted in CMake
static const std::string           kFilename = "single_palm.pcd";

int main() {
    try {
        const std::filesystem::path in_path  = kDataDir / kFilename;
        const std::filesystem::path out_dir  = kDataDir / "cloud_out";
        const std::filesystem::path out_path = out_dir / "single_palm_xyz_normals.pcd";
        std::filesystem::create_directories(out_dir); // ensure cloud_out exists

        std::cout << "Running normal extraction\n"
                  << "Opening " << in_path << "\n";

        // --- Load as generic container (PCLPointCloud2) ---
        pcl::PCLPointCloud2 cloud_blob;
        if (pcl::io::loadPCDFile(in_path.string(), cloud_blob) < 0) {
            std::cerr << "Failed to read " << in_path << "\n";
            return 1;
        }

        pcl::console::TicToc tt;

        // --- Convert to typed XYZ cloud ---
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromPCLPointCloud2(cloud_blob, *cloud_xyz);
        if (cloud_xyz->empty()) {
            std::cerr << "Loaded cloud has 0 points (no XYZ fields?)\n";
            return 1;
        }
        std::cout << "Loaded " << cloud_xyz->size() << " points\n";

        // --- Voxelisation downsampling ---
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxelised(new pcl::PointCloud<pcl::PointXYZ>);
        std::cerr << "Downsampling...\n";
        tt.tic();
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud_xyz);
        vg.setLeafSize(0.03f, 0.03f, 0.03f);
        vg.setDownsampleAllData(true);
        vg.filter(*cloud_voxelised);
        std::cerr << "Voxelised: " << cloud_voxelised->size()
                  << " points in " << tt.toc() << " ms\n";

        // --- KD-tree + Normal estimation (on voxelised cloud) ---
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud_voxelised);

        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setViewPoint(0.f, 0.f, 1.f);          // optional: orient normals
        ne.setInputCloud(cloud_voxelised);          // << was cloud_xyz
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.03);                   // 3 cm; or use ne.setKSearch(20);
        ne.compute(*cloud_normals);
        std::cout << "Number of normals: " << cloud_normals->size() << "\n";

        if (cloud_normals->size() != cloud_voxelised->size()) { // << check against voxelised
            std::cerr << "Normals/points size mismatch\n";
            return 2;
        }

        // --- Concatenate XYZ + normals (voxelised) for saving ---
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
        pcl::concatenateFields(*cloud_voxelised, *cloud_normals, *cloud_pn); // << voxelised

        if (pcl::io::savePCDFileBinary(out_path.string(), *cloud_pn) < 0) {
            std::cerr << "Failed to write " << out_path << "\n";
            return 3;
        }
        std::cout << "Wrote " << out_path << " (" << cloud_pn->size() << " points)\n";

        // --- Visualize voxelised points + subset of normals ---
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Normals Viewer"));
        viewer->addCoordinateSystem(1.0);
        viewer->setBackgroundColor(0, 0, 0);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white(cloud_voxelised, 255, 255, 255);
        viewer->addPointCloud<pcl::PointXYZ>(cloud_voxelised, white, "cloud"); // << voxelised
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

        const int level = 3;      // draw every Nth normal
        const double scale = 0.05;  // arrow length (meters)
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_voxelised, cloud_normals, level, scale, "normals"); // << voxelised
        viewer->initCameraParameters();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 4;
    }
}
