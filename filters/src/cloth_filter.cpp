#include <iostream>
#include <filesystem>
#include <limits>
#include <cmath>
#include <chrono>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>                 // pcl::getFieldsList
#include <pcl/filters/extract_indices.h>   // split by indices

// CSF
#include <CSF.h>            // installed in ~/.local/include via your setup

// Same macro style you used
static const std::filesystem::path dataDir = DATA_DIR;  // must be a quoted macro
static const std::string filename = "forest3.pcd";

int main()
{
    using PointT = pcl::PointXYZ;

    const std::filesystem::path in_path   = dataDir / filename;
    const std::filesystem::path out_dir   = dataDir / "cloud_out";
    const std::filesystem::path out_ground= out_dir / "forest3_ground_only.pcd";
    const std::filesystem::path out_nong  = out_dir / "forest3_no_ground.pcd";
    std::filesystem::create_directories(out_dir);

    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    if (reader.read(in_path.string(), *cloud) < 0) {
        std::cerr << "Failed to read " << in_path << "\n";
        return 1;
    }

    std::cerr << "Loaded: " << cloud->width * cloud->height
              << " points (" << pcl::getFieldsList(*cloud) << ")\n";

    auto start = std::chrono::high_resolution_clock::now();

    // ---- Convert PCL -> CSF simple point list ----
    // AFTER (right)
    csf::PointCloud csf_points; // vector of points 
    std::vector<int> idx_map;
    idx_map.reserve(cloud->size());

    for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
        const auto& p = cloud->points[i];
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            csf_points.push_back(csf::Point(p.x, p.y, p.z));
            idx_map.push_back(i); // map CSF index -> original PCL index
        }
    }
    if (csf_points.empty()) { std::cerr << "No finite points.\n"; return 1; }

    // ---- Configure CSF ----
    CSF csf;
    csf.setPointCloud(csf_points);

    // Params: good starters for UAV LiDAR over oil-palm/forest
    csf.params.bSloopSmooth    = false;   // keep cloth from excessive smoothing on slopes
    csf.params.cloth_resolution= 0.5f;    // meters; 0.5–1.0 typical
    csf.params.rigidness       = 1;       // 2–4; higher = stiffer cloth
    csf.params.time_step       = 0.65f;   // default is fine; leave unless unstable
    csf.params.class_threshold = 0.45f;   // distance to cloth considered ground (m)

    // ---- Run filter ----
    std::vector<int> ground_idx, nonground_idx;
    try {
        
        csf.do_filtering(ground_idx, nonground_idx);
    } catch (const std::exception &e) {
        std::cerr << "CSF filtering failed: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "CSF done. Ground: " << ground_idx.size()
              << "  Non-ground: " << nonground_idx.size() << "\n";

    // ---- Map CSF indices back to original cloud ----
    // CSF preserved input order, so indices line up with 'cloud'
    pcl::PointIndices::Ptr ground_pi(new pcl::PointIndices);
    ground_pi->indices.reserve(ground_idx.size());
    for (int k : ground_idx) ground_pi->indices.push_back(idx_map[k]);

    pcl::PointIndices::Ptr nong_pi(new pcl::PointIndices);
    nong_pi->indices.reserve(nonground_idx.size());
    for (int k : nonground_idx) nong_pi->indices.push_back(idx_map[k]);

    // ---- Extract and write ----
    pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr nong_cloud(new pcl::PointCloud<PointT>);

    pcl::ExtractIndices<PointT> ex;
    ex.setInputCloud(cloud);

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


    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Elapsed time: " << duration_ms.count() << " milliseconds" << std::endl;

    if (writer.write(out_ground.string(), *ground_cloud, false)) {
        std::cerr << "Failed to write " << out_ground << "\n";
        return 1;
    }
    if (writer.write(out_nong.string(), *nong_cloud, false)) {
        std::cerr << "Failed to write " << out_nong << "\n";
        return 1;
    }

    std::cout << "Wrote:\n  " << out_ground << "  (" << ground_cloud->size() << " pts)\n"
              << "  " << out_nong   << "  (" << nong_cloud->size()   << " pts)\n";

    return 0;
}
