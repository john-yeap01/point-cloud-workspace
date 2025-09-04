// add CSF before clustering

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h> // removeNaNFromPointCloud
#include <pcl/common/io.h>      // pcl::getFieldsList

#include <iomanip>
#include <filesystem>
#include <random>
#include <chrono>
#include <iostream>
#include <limits>
#include <cmath>
#include <vector>

// === CSF ===
#include <CSF.h>   // make sure your include path + linker path are set

// I/O roots (expects CMake to -D DATA_DIR=".../data")
static const std::filesystem::path dataDir = DATA_DIR;
// static const std::string filename = "sectionA.pcd";  // <-- change if needed
static const std::string filename = "forest3.pcd"; 

int main()
{
    using PointT = pcl::PointXYZ;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Paths
    const std::filesystem::path in_path  = dataDir / filename;
    const std::filesystem::path out_dir  = dataDir / "cloud_out";
    const std::filesystem::path clu_dir  = out_dir / "clusters";
    std::filesystem::create_directories(clu_dir);

    // Read input
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    if (reader.read(in_path.string(), *cloud) < 0) {
        PCL_ERROR("Couldn't read file %s\n", in_path.string().c_str());
        return -1;
    }
    std::cout << "Loaded: " << cloud->size()
              << " pts (" << pcl::getFieldsList(*cloud) << ")\n";


    auto so = cloud->sensor_origin_;
    auto sq = cloud->sensor_orientation_;
    std::cout << "base origin: " << so.transpose()
    << "  base quat: " << sq.coeffs().transpose() << "\n";

    // Optional: quick downsample to speed up CSF & clustering
    pcl::PointCloud<PointT>::Ptr cloud_ds(new pcl::PointCloud<PointT>);
    {
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(cloud);
        // 1 cm (0.01 m) was your original; bump to 2–3 cm if you want faster
        vg.setLeafSize(0.05f, 0.05f, 0.05f);
        vg.filter(*cloud_ds);
    }
    std::cout << "After VoxelGrid: " << cloud_ds->size() << " pts\n";

    // -------- CSF ground / non-ground split --------
    // Build CSF point list (skip NaNs here to avoid index mismatches)
    csf::PointCloud csf_points;
    csf_points.reserve(cloud_ds->size());
    std::vector<int> idx_map; idx_map.reserve(cloud_ds->size()); // CSF idx -> cloud_ds idx

    for (int i = 0; i < static_cast<int>(cloud_ds->size()); ++i) {
        const auto& p = (*cloud_ds)[i];
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            csf_points.emplace_back(p.x, p.y, p.z);
            idx_map.push_back(i);
        }
    }
    if (csf_points.empty()) {
        std::cerr << "No finite points for CSF.\n";
        return 1;
    }

    CSF csf;
    csf.setPointCloud(csf_points);

    // Params tuned for plantation/UAV/Terrestrial mixes; tweak once:
    csf.params.bSloopSmooth     = false;   // keep from over-smoothing slopes
    csf.params.cloth_resolution = 0.5f;    // 0.5–1.0 m typical; smaller hugs micro-terrain more
    csf.params.rigidness        = 2;       // 1–4; higher = stiffer cloth
    csf.params.time_step        = 0.65f;   // default fine
    csf.params.class_threshold  = 0.45f;   // dist to cloth considered ground (m)

    std::vector<int> ground_idx, nonground_idx;
    try {
        csf.do_filtering(ground_idx, nonground_idx);
    } catch (const std::exception& e) {
        std::cerr << "CSF failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << "CSF: ground=" << ground_idx.size()
              << " nonground=" << nonground_idx.size() << "\n";

    // Map CSF indices back to cloud_ds indices
    pcl::PointIndices::Ptr ground_pi(new pcl::PointIndices);
    ground_pi->indices.reserve(ground_idx.size());
    for (int k : ground_idx) ground_pi->indices.push_back(idx_map[k]);

    pcl::PointIndices::Ptr nong_pi(new pcl::PointIndices);
    nong_pi->indices.reserve(nonground_idx.size());
    for (int k : nonground_idx) nong_pi->indices.push_back(idx_map[k]);

    // Extract clouds
    pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr nong_cloud(new pcl::PointCloud<PointT>);
    {
        pcl::ExtractIndices<PointT> ex;
        ex.setInputCloud(cloud_ds);

        if (!ground_pi->indices.empty()) {
            ex.setIndices(ground_pi);
            ex.setNegative(false);
            ex.filter(*ground_cloud);
        }
        if (!nong_pi->indices.empty()) {
            ex.setIndices(nong_pi);
            ex.setNegative(false);
            ex.filter(*nong_cloud);
        }
    }

    // (Optional) Write ground/non-ground
    const auto out_ground = out_dir / "ground_only_csf.pcd";
    const auto out_nong   = out_dir / "no_ground_csf.pcd";
    writer.write(out_ground.string(), *ground_cloud, false);
    writer.write(out_nong.string(),   *nong_cloud,   false);
    std::cout << "Wrote:\n  " << out_ground << " (" << ground_cloud->size() << " pts)\n"
              << "  " << out_nong   << " (" << nong_cloud->size()   << " pts)\n";

    // -------- Clean NaNs (safety) --------
    std::vector<int> keep;
    pcl::PointCloud<PointT>::Ptr cloud_no_nan(new pcl::PointCloud<PointT>);
    pcl::removeNaNFromPointCloud(*nong_cloud, *cloud_no_nan, keep);
    if (cloud_no_nan->empty()) {
        std::cerr << "All non-ground points were NaN?\n";
        return 1;
    }
    std::cout << "Removed " << (nong_cloud->size() - cloud_no_nan->size()) << " NaNs\n";

    // -------- Euclidean clustering on non-ground --------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_no_nan);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.1);   // 0.10 m = 10 cm (your comment said 2 cm earlier)
    ec.setMinClusterSize(10000);     // adjust to your density; you had 10000 (quite high)
    ec.setMaxClusterSize(500000);   // allow large crowns
    // ec.setMaxClusterSize(200000); 
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_no_nan);
    ec.extract(cluster_indices);

    std::cout << "Clusters found: " << cluster_indices.size() << "\n";

    // Save each cluster and prepare viz
    std::vector<pcl::PointCloud<PointT>::Ptr> clusters;
    clusters.reserve(cluster_indices.size());

    pcl::PCDWriter w;
    int j = 0;
    for (const auto& ci : cluster_indices) {
        pcl::PointCloud<PointT>::Ptr C(new pcl::PointCloud<PointT>);
        C->reserve(ci.indices.size());
        for (int idx : ci.indices) C->push_back((*cloud_no_nan)[idx]);
        C->width = C->size(); C->height = 1; C->is_dense = true;

        std::stringstream ss; ss << std::setw(4) << std::setfill('0') << j;
        const auto out_path = clu_dir / ("cloud_cluster_" + ss.str() + ".pcd");
        w.write<PointT>(out_path.string(), *C, false);
        std::cout << "Cluster " << j << ": " << C->size() << " pts → " << out_path << "\n";
        clusters.push_back(C);
        ++j;
    }

    // -------- Visualization --------
    if (!clusters.empty()) {
        pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer("CSF→Clusters"));
        vis->setBackgroundColor(0, 0, 0);

        // Show base (non-ground) in grey
        pcl::visualization::PointCloudColorHandlerCustom<PointT> base_color(cloud_no_nan, 140, 140, 140);
        vis->addPointCloud<PointT>(cloud_no_nan, base_color, "base");
        vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "base");

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> U(40, 235);
        for (size_t i = 0; i < clusters.size(); ++i) {
            auto& C = clusters[i];
            int r = U(rng), g = U(rng), b = U(rng);
            std::string id = "cluster_" + std::to_string(i);
            pcl::visualization::PointCloudColorHandlerCustom<PointT> clr(C, r, g, b);
            vis->addPointCloud<PointT>(C, clr, id);
            vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, id);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Elapsed time: " << elapsed.count() << " s\n";

        while (!vis->wasStopped()) vis->spinOnce(16);
    } else {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "No clusters found. Elapsed time: " << elapsed.count() << " s\n";
    }

    return 0;
}
