// trunks_from_stumps.cpp
// 1) Slice z in [0.5, 1.5] and cluster → stump candidates
// 2) For each stump centroid (x,y), select a cylinder (~1m radius) from the FULL cloud
// 3) Cluster points inside each cylinder and keep the largest cluster as the trunk
// 4) Save per-trunk clouds and overlay everything for visualization

#include <iostream>
#include <filesystem>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>     // removeNaNFromPointCloud
#include <pcl/common/common.h>      // getMinMax3D

static const std::filesystem::path dataDir = DATA_DIR;  // must be a quoted macro
static const std::string filename = "sectionA.pcd";

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

struct Params {
    // Stump slice band (meters)
    float slice_z_min = 0.1f;
    float slice_z_max = 0.8f;

    // Clustering on the slice (to find stump candidates)
    float slice_cluster_tolerance = 0.12f; // depends on point spacing
    int   slice_min_cluster_size   = 300;  // tune with your density
    int   slice_max_cluster_size   = 250000;

    // Cylinder selection around stump centroids, on the full cloud
    float cylinder_radius = 0.3f;    // “additional offset about 0.3 m radius”
    float cyl_z_min       = 0.1f;    // expand if you want taller trunks
    float cyl_z_max       = 10.0f;

    // Clustering inside each cylinder to isolate the trunk
    float trunk_cluster_tolerance = 0.08f;
    int   trunk_min_cluster_size   = 400;
    int   trunk_max_cluster_size   = 400000;

    // Downsampling (optional)
    bool  downsample_slice = false;
    float slice_leaf        = 0.03f;

    bool  downsample_cylinders = false;
    float cyl_leaf             = 0.03f;

    // Viz
    int base_point_size  = 2;
    int trunk_point_size = 6;
};

static CloudT::Ptr indicesToCloud(const CloudT::ConstPtr& src, const pcl::PointIndices& idxs)
{
    CloudT::Ptr out(new CloudT);
    out->reserve(idxs.indices.size());
    for (int i : idxs.indices) out->push_back((*src)[i]);
    out->width = out->size(); out->height = 1; out->is_dense = true;
    return out;
}

static CloudT::Ptr largestEuclideanCluster(const CloudT::ConstPtr& pc,
                                           float tol, int min_sz, int max_sz)
{
    CloudT::Ptr empty(new CloudT);
    if (!pc || pc->empty()) return empty;

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(pc);

    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(tol);
    ec.setMinClusterSize(min_sz);
    ec.setMaxClusterSize(max_sz);
    ec.setSearchMethod(tree);
    ec.setInputCloud(pc);
    ec.extract(clusters);

    if (clusters.empty()) return empty;

    const pcl::PointIndices* best = &clusters.front();
    for (const auto& c : clusters) if (c.indices.size() > best->indices.size()) best = &c;
    return indicesToCloud(pc, *best);
}

// brute force XY cylinder pick from a source cloud, with Z band
static CloudT::Ptr pickCylinder(const CloudT::ConstPtr& src,
                                const Eigen::Vector2f& cxy,
                                float radius, float zmin, float zmax)
{
    CloudT::Ptr out(new CloudT);
    if (!src || src->empty()) return out;

    const float r2 = radius*radius;
    out->reserve(4096);
    for (const auto& p : *src) {
        if (p.z < zmin || p.z > zmax) continue;
        const float dx = p.x - cxy.x();
        const float dy = p.y - cxy.y();
        if (dx*dx + dy*dy <= r2) out->push_back(p);
    }
    out->width = out->size(); out->height = 1; out->is_dense = true;
    return out;
}

// compute XY centroid of a cloud (ignore Z)
static Eigen::Vector2f centroidXY(const CloudT::ConstPtr& pc)
{
    double sx=0, sy=0; size_t n=0;
    for (const auto& p : *pc) { sx += p.x; sy += p.y; ++n; }
    if (n == 0) return Eigen::Vector2f(0.f, 0.f);
    return Eigen::Vector2f(static_cast<float>(sx/n), static_cast<float>(sy/n));
}

int main(int, char**)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    const Params P;

    const std::filesystem::path in_path  = dataDir / filename;
    const std::filesystem::path out_dir  = dataDir / "cloud_out";
    const std::filesystem::path stump_dir= out_dir / "stumps_slice_clusters";
    const std::filesystem::path trunk_dir= out_dir / "trunks_from_cylinders";
    std::filesystem::create_directories(stump_dir);
    std::filesystem::create_directories(trunk_dir);

    pcl::PCDReader reader; pcl::PCDWriter writer;

    // ---- Load full cloud
    CloudT::Ptr cloud(new CloudT);
    if (reader.read(in_path.string(), *cloud) != 0) {
        std::cerr << "Failed to read: " << in_path << "\n";
        return 1;
    }
    std::vector<int> dummy; pcl::removeNaNFromPointCloud(*cloud, *cloud, dummy);
    std::cout << "Loaded " << cloud->size() << " pts: \"" << in_path.string() << "\"\n";

    // ---- Slice z in [0.5, 1.5]
    CloudT::Ptr slice(new CloudT);
    {
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(P.slice_z_min, P.slice_z_max);
        pass.filter(*slice);
    }
    std::cout << "Stump slice z ∈ [" << P.slice_z_min << ", " << P.slice_z_max
              << "] → " << slice->size() << " pts\n";

    if (slice->empty()) {
        std::cerr << "Slice is empty; adjust z band.\n";
        return 0;
    }

    // Optional: downsample slice
    if (P.downsample_slice) {
        CloudT::Ptr ds(new CloudT);
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(slice);
        vg.setLeafSize(P.slice_leaf, P.slice_leaf, P.slice_leaf);
        vg.filter(*ds);
        std::cout << "Downsampled slice → " << ds->size() << " pts\n";
        slice.swap(ds);
    }

    // XY bounds of slice for sanity
    PointT minP, maxP;
    pcl::getMinMax3D(*slice, minP, maxP);
    std::cout << std::fixed << std::setprecision(3)
              << "Slice XY bounds: X[" << minP.x << "," << maxP.x
              << "]  Y[" << minP.y << "," << maxP.y
              << "]  Z[" << minP.z << "," << maxP.z << "]\n";

    // ---- Cluster the slice to get stump candidates
    pcl::search::KdTree<PointT>::Ptr slice_tree(new pcl::search::KdTree<PointT>);
    slice_tree->setInputCloud(slice);

    std::vector<pcl::PointIndices> slice_clusters_idx;
    {
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(P.slice_cluster_tolerance);
        ec.setMinClusterSize(P.slice_min_cluster_size);
        ec.setMaxClusterSize(P.slice_max_cluster_size);
        ec.setSearchMethod(slice_tree);
        ec.setInputCloud(slice);
        ec.extract(slice_clusters_idx);
    }
    std::cout << "Slice clusters (stump candidates): " << slice_clusters_idx.size() << "\n";

    // Save stump clusters & collect centroids
    std::vector<Eigen::Vector2f> stump_centers_xy;
    stump_centers_xy.reserve(slice_clusters_idx.size());

    {
        int i = 0;
        for (const auto& ci : slice_clusters_idx) {
            auto C = indicesToCloud(slice, ci);
            auto ctr = centroidXY(C);
            stump_centers_xy.push_back(ctr);

            std::stringstream ss; ss << std::setw(4) << std::setfill('0') << i;
            const auto out_path = stump_dir / ("stump_slice_cluster_" + ss.str() + ".pcd");
            writer.write<PointT>(out_path.string(), *C, false);
            std::cout << "Stump " << i << ": " << C->size() << " pts, ctr=("
                      << ctr.x() << "," << ctr.y() << ") → " << out_path << "\n";
            ++i;
        }
    }

    // ---- For each stump center, pick cylinder in FULL cloud, then cluster → trunk
    std::vector<CloudT::Ptr> trunks;
    trunks.reserve(stump_centers_xy.size());

    int ti = 0;
    for (const auto& cxy : stump_centers_xy) {
        auto cyl = pickCylinder(cloud, cxy, P.cylinder_radius, P.cyl_z_min, P.cyl_z_max);
        std::cout << "Center " << ti << " (" << cxy.x() << "," << cxy.y()
                  << ") → cylinder pts: " << cyl->size() << "\n";

        if (P.downsample_cylinders && !cyl->empty()) {
            CloudT::Ptr ds(new CloudT);
            pcl::VoxelGrid<PointT> vg;
            vg.setInputCloud(cyl);
            vg.setLeafSize(P.cyl_leaf, P.cyl_leaf, P.cyl_leaf);
            vg.filter(*ds);
            cyl.swap(ds);
            std::cout << "  cylinder downsampled → " << cyl->size() << " pts\n";
        }

        auto trunk = largestEuclideanCluster(cyl,
                                             P.trunk_cluster_tolerance,
                                             P.trunk_min_cluster_size,
                                             P.trunk_max_cluster_size);
        trunks.push_back(trunk);

        std::stringstream ss; ss << std::setw(4) << std::setfill('0') << ti;
        const auto out_path = trunk_dir / ("trunk_" + ss.str() + ".pcd");
        if (trunk && !trunk->empty()) {
            writer.write<PointT>(out_path.string(), *trunk, false);
            std::cout << "  trunk " << ti << ": " << trunk->size()
                      << " pts → " << out_path << "\n";
        } else {
            std::cout << "  trunk " << ti << ": (empty)\n";
        }
        ++ti;
    }

    // Merge all trunks for convenience
    CloudT::Ptr trunks_merged(new CloudT);
    for (const auto& t : trunks) if (t) *trunks_merged += *t;
    const auto merged_path = trunk_dir / "trunks_merged.pcd";
    if (!trunks_merged->empty())
        writer.write<PointT>(merged_path.string(), *trunks_merged, false);
    std::cout << "Merged trunks: " << trunks_merged->size()
              << " pts → " << merged_path << "\n";

    // ---- Visualization: base + trunks + (optional) stump centers
    pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer("Trunks from Stumps"));
    vis->setBackgroundColor(0, 0, 0);

    // base cloud (grey)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT>
            base_color(cloud, 140, 140, 140);
        vis->addPointCloud<PointT>(cloud, base_color, "base");
        vis->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, P.base_point_size, "base");
    }

    // Draw stump center cues
    for (size_t i = 0; i < stump_centers_xy.size(); ++i) {
        const auto& cxy = stump_centers_xy[i];
        PointT a, b; a.x = cxy.x(); a.y = cxy.y(); a.z = P.slice_z_min;
        b.x = cxy.x(); b.y = cxy.y(); b.z = P.slice_z_max;
        std::string lid = "center_line_" + std::to_string(i);
        vis->addLine(a, b, 255, 255, 0, lid);
        PointT s; s.x = cxy.x(); s.y = cxy.y(); s.z = 0.5f*(P.slice_z_min+P.slice_z_max);
        std::string sid = "center_sphere_" + std::to_string(i);
        vis->addSphere(s, 0.08, 255, 255, 0, sid);
    }

    // random colors for trunks
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> U(40, 235);

    for (size_t i = 0; i < trunks.size(); ++i) {
        auto& T = trunks[i];
        if (!T || T->empty()) continue;
        int r = U(rng), g = U(rng), b = U(rng);
        std::string id = "trunk_" + std::to_string(i);

        pcl::visualization::PointCloudColorHandlerCustom<PointT> clr(T, r, g, b);
        vis->addPointCloud<PointT>(T, clr, id);
        vis->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, P.trunk_point_size, id);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    std::cout << "Showing overlay. Press 'q' to close.\n";
    while (!vis->wasStopped()) vis->spinOnce(16);
    return 0;
}
