#include <iostream>
#include <string>
#include <vector>
#include <cmath>  // std::cos/sin

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

static bool loadCloud(const std::string& path, Cloud::Ptr cloud) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".pcd") {
        return pcl::io::loadPCDFile(path, *cloud) == 0;
    } else if (path.size() >= 4 && path.substr(path.size() - 4) == ".ply") {
        return pcl::io::loadPLYFile(path, *cloud) == 0;
    } else {
        std::cerr << "Unsupported file extension (use .pcd or .ply)\n";
        return false;
    }
}

int main(int argc, char** argv) {
    Cloud::Ptr cloud(new Cloud);

    if (argc >= 2) {
        if (!loadCloud(argv[1], cloud)) {
            std::cerr << "Failed to load: " << argv[1] << "\n";
            return 1;
        }
        std::cout << "Loaded " << cloud->size() << " points from " << argv[1] << "\n";
    } else {
        // Synthetic sample: plane (z=0) + small circular blob
        cloud->reserve(2000);
        for (int i = 0; i < 1500; ++i)
            cloud->push_back(pcl::PointXYZ(float(i % 50) * 0.01f, float(i / 50) * 0.01f, 0.0f));
        for (int i = 0; i < 500; ++i) {
            float angle = static_cast<float>(i) * 0.04f;
            float cx = 0.5f + 0.05f * static_cast<float>(std::cos(static_cast<double>(angle)));
            float cy = 0.5f + 0.05f * static_cast<float>(std::sin(static_cast<double>(angle)));
            cloud->push_back(pcl::PointXYZ(cx, cy, 0.05f));
        }
        cloud->width = static_cast<uint32_t>(cloud->size());
        cloud->height = 1;
        cloud->is_dense = true;
        std::cout << "Generated synthetic cloud: " << cloud->size() << " points\n";
    }

    // 1) Voxel downsample
    const float voxel = 0.02f; // 2 cm
    Cloud::Ptr ds(new Cloud);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*ds);
    std::cout << "Downsampled: " << ds->size() << " points\n";

    // 2) RANSAC plane
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(std::max(0.5f * voxel, 0.005f));
    seg.setInputCloud(ds);

    pcl::ModelCoefficients coeff;
    pcl::PointIndices inliers;
    seg.segment(inliers, coeff);
    std::cout << "Plane inliers: " << inliers.indices.size() << "\n";

    // Extract ground/rest — use PCL smart pointers (no Boost make_shared)
    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices(inliers));
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(ds);
    extract.setIndices(inliers_ptr);

    Cloud::Ptr ground(new Cloud), rest(new Cloud);
    extract.setNegative(false); extract.filter(*ground);
    extract.setNegative(true);  extract.filter(*rest);
    std::cout << "Remaining points (no ground): " << rest->size() << "\n";

    // 3) Euclidean clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(rest);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(std::max(1.5f * voxel, 0.01f));
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(250000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(rest);
    ec.extract(cluster_indices);

    std::cout << "Found clusters: " << cluster_indices.size() << "\n";

    // 4) Save outputs
    pcl::io::savePCDFileBinary("ground.pcd", *ground);
    pcl::io::savePCDFileBinary("rest.pcd", *rest);
    int id = 0;
    for (const auto& idx : cluster_indices) {
        Cloud::Ptr cluster(new Cloud);
        cluster->reserve(idx.indices.size());
        for (int i : idx.indices) cluster->push_back((*rest)[i]);
        cluster->width = static_cast<uint32_t>(cluster->size());
        cluster->height = 1;
        cluster->is_dense = true;
        std::string out = "cluster_" + std::to_string(id++) + ".pcd";
        pcl::io::savePCDFileBinary(out, *cluster);
        std::cout << "Saved " << out << " (" << cluster->size() << " pts)\n";
    }

    std::cout << "Done.\n";
    return 0;
}
