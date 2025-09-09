// ransac_trunks.cpp
#include <iostream>
#include <filesystem>
#include <cmath>

#include "viewer.hpp"  // basics::makeViewer(cloud, title)

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointT;

template <class Ptr>
static void log_count(const char* tag, const Ptr& pc){ std::cerr << tag << ": " << pc->size() << " pts\n"; }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const std::filesystem::path dataDir = DATA_DIR; // injected by CMake
static const std::string filename = "forest3.pcd";

static inline float deg2rad(float d){ return d * float(M_PI) / 180.0f; }



int main(){
    std::cout << "Tree trunk finder (RANSAC cylinders)\n";
    std::cout << "试图中文" << std::endl;

    auto rad2deg = [](double r){ return r * 180.0 / M_PI; };

    const int    MAX_TRUNKS     = 10;      // stop after N cylinders
    const int    MIN_INLIERS    = 70;    // reject tiny fits
    const double MAX_TILT_DEG   = 20.0;   // “vertical” tolerance
    const double NORMAL_RADIUS  = 0.10;   // ~3–4× your voxel (0.03 m)

    

    const auto in_path = dataDir / filename;
    const auto out_dir = dataDir / "cloud_out";
    std::filesystem::create_directories(out_dir);

    // ---- PCL objects ----
    pcl::PCDReader reader;  pcl::PCDWriter writer;
    pcl::VoxelGrid<PointT> vg;
    pcl::PassThrough<PointT> band;
    pcl::RadiusOutlierRemoval<PointT> ror;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>());
    pcl::ExtractIndices<PointT> ex_pts;
    pcl::ExtractIndices<pcl::Normal> ex_nrm;
    pcl::PointCloud<PointT>::Ptr ground (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr no_ground(new pcl::PointCloud<PointT>);

    pcl::ModelCoefficients::Ptr coeff_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

    // Ground remover: plane ⟂ Z
    pcl::SACSegmentation<PointT> seg_ground;
    seg_ground.setOptimizeCoefficients(true);
    seg_ground.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg_ground.setMethodType(pcl::SAC_RANSAC);
    seg_ground.setAxis(Eigen::Vector3f(0,0,1));
    seg_ground.setEpsAngle(deg2rad(10.0f));
    seg_ground.setMaxIterations(500);
    seg_ground.setDistanceThreshold(0.03);


    // CYLINDER 
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg_cyl;
    pcl::PointIndices::Ptr inliers_cyl(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff_cyl(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // ---- Cylinder RANSAC config (do once) ----
    seg_cyl.setOptimizeCoefficients(true);
    seg_cyl.setModelType(pcl::SACMODEL_CYLINDER);
    seg_cyl.setMethodType(pcl::SAC_RANSAC);
    seg_cyl.setNormalDistanceWeight(0.05);
    seg_cyl.setMaxIterations(6000);
    seg_cyl.setDistanceThreshold(0.025);      // ~ voxel size (0.02–0.03)
    seg_cyl.setRadiusLimits(0.06, 0.35);      // tighten to your trunks

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_voxel(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_band(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cyl_pts(new pcl::PointCloud<PointT>);

    // ---- Load ----
    std::cout << "Reading: " << in_path << "\n";
    if (reader.read(in_path.string(), *cloud) < 0) { PCL_ERROR("Couldn't read file\n"); return -1; }
    std::cerr << "Raw points: " << cloud->size() << "\n";

    // 1) Height band FIRST (on raw cloud)
    pcl::PointCloud<PointT>::Ptr cloud_band_raw(new pcl::PointCloud<PointT>);
    band.setInputCloud(cloud);
    band.setFilterFieldName("z");
    band.setFilterLimits(0.3, 4.0);            // widen for now
    band.filter(*cloud_band_raw);
    log_count("band_raw",           cloud_band_raw);

    // 2) THEN voxelize the already-cropped cloud
    vg.setInputCloud(cloud_band_raw);
    vg.setLeafSize(0.03f, 0.03f, 0.03f);       // 3 cm
    vg.filter(*cloud_voxel);
    log_count("voxel",              cloud_voxel);

    // ---- De-spike ----
    ror.setInputCloud(cloud_voxel);
    ror.setRadiusSearch(0.15f);
    ror.setMinNeighborsInRadius(4);
    ror.filter(*cloud_band);
    log_count("after_ROR",          cloud_band);

    // ---- Ground removal ----
    seg_ground.setInputCloud(cloud_band);
    seg_ground.segment(*inliers_plane, *coeff_plane);
    std::cerr << "Ground plane: " << *coeff_plane << "\n";

    if (inliers_plane->indices.empty()) {
        std::cerr << "No ground plane found.\n";
        return 0; // or skip ground removal
    }

    ex_pts.setInputCloud(cloud_band);
    ex_pts.setIndices(inliers_plane);
    ex_pts.setNegative(false);
    ex_pts.filter(*ground);
    log_count("ground_inliers",     ground);

    ex_pts.setNegative(true);
    ex_pts.filter(*no_ground);
    log_count("no_ground",          no_ground);

    // ---- Viewer ----
    auto vis = basics::makeViewer(no_ground, "Cloud band raw");

    pcl::PointCloud<PointT>::Ptr work(new pcl::PointCloud<PointT>(*no_ground));
    for (int t = 0; t < MAX_TRUNKS; ++t) {
        if (work->size() < 200) break;  // too sparse to continue

        // 1) Normals on current work cloud
        pcl::PointCloud<pcl::Normal>::Ptr norms(new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud(work);
        ne.setSearchMethod(kdtree);
        ne.setRadiusSearch(NORMAL_RADIUS);   // or ne.setKSearch(20);
        ne.compute(*norms);
        std::cerr << "[t" << t << "] work:" << work->size()
          << " normals:" << norms->size() << "\n";

        // 2) Segment cylinder
        pcl::PointIndices::Ptr inliers_cyl(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coeff_cyl(new pcl::ModelCoefficients);
        // inputs + constraints (before segment)
        seg_cyl.setAxis(Eigen::Vector3f(0,0,1));
        seg_cyl.setEpsAngle(deg2rad(MAX_TILT_DEG));
        seg_cyl.setInputCloud(work);
        seg_cyl.setInputNormals(norms);

        // run
        seg_cyl.segment(*inliers_cyl, *coeff_cyl);


        if (inliers_cyl->indices.size() < MIN_INLIERS) {
            std::cerr << "[t" << t << "] stop: not enough inliers (" 
                    << inliers_cyl->indices.size() << ")\n";
            break;
        }

        // 3) (Optional) extra tilt check/print
        const auto& c = coeff_cyl->values; // [x0,y0,z0, ax,ay,az, r]
        double ax=c[3], ay=c[4], az=c[5];
        double tilt = rad2deg(std::acos(std::abs(az) / std::sqrt(ax*ax + ay*ay + az*az)));
        std::cerr << "[t" << t << "] r=" << c[6] << " m, tilt≈" << tilt 
                << "°, inliers=" << inliers_cyl->indices.size() << "\n";
        if (tilt > MAX_TILT_DEG) {
            // Shouldn’t happen because of setAxis+setEpsAngle, but keep as guard.
            std::cerr << "  rejected (tilt)\n";
            // Option A: remove these inliers anyway to avoid re-picking same bad patch:
            pcl::PointCloud<PointT>::Ptr tmp(new pcl::PointCloud<PointT>);
            ex_pts.setInputCloud(work); ex_pts.setIndices(inliers_cyl);
            ex_pts.setNegative(true); ex_pts.filter(*tmp);
            work.swap(tmp);
            continue;
        }

        // 4) Extract the cylinder points
        pcl::PointCloud<PointT>::Ptr cyl_pts(new pcl::PointCloud<PointT>);
        ex_pts.setInputCloud(work);
        ex_pts.setIndices(inliers_cyl);
        ex_pts.setNegative(false);
        ex_pts.filter(*cyl_pts);

        // Save each trunk
        writer.write<PointT>((out_dir / ("trunk_" + std::to_string(t) + ".pcd")).string(),
                            *cyl_pts, false);

        // Overlay in viewer (green) — you can reuse one viewer outside the loop
        // (Assuming you already created 'vis' from no_ground before the loop)
        {
            pcl::visualization::PointCloudColorHandlerCustom<PointT> green(cyl_pts, 0, 255, 0);
            const std::string id = "trunk" + std::to_string(t);
            if (!vis->updatePointCloud<PointT>(cyl_pts, green, id)) {
                vis->addPointCloud<PointT>(cyl_pts, green, id);
            }
            vis->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, id);
        }

        // 5) Peel: remove inliers from 'work' and continue
        pcl::PointCloud<PointT>::Ptr remaining(new pcl::PointCloud<PointT>);
        ex_pts.setNegative(true);
        ex_pts.filter(*remaining);
        work.swap(remaining);
    }


    

    writer.write<PointT>((out_dir / "band_after_ror.pcd").string(), *cloud_band, false);
    writer.write<PointT>((out_dir / "ground_plane.pcd").string(),   *ground,    false);
    writer.write<PointT>((out_dir / "no_ground.pcd").string(),      *no_ground, false);

    // // Overlay the cylinder points in green
    // if (!inliers_cyl->indices.empty()) {
    //     pcl::visualization::PointCloudColorHandlerCustom<PointT> green(cyl_pts, 0, 255, 0);

    //     // Reuse the existing viewer (created below), or create it now if you prefer
    //     // auto vis = basics::makeViewer(no_ground, "No ground");  // you already have this later

    //     // If the id exists, update; otherwise, add once
    //     if (!vis->updatePointCloud<PointT>(cyl_pts, green, "trunk0")) {
    //         vis->addPointCloud<PointT>(cyl_pts, green, "trunk0");
    //     }
    //     vis->setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "trunk0");
    // }

    if (cloud_band->points.empty())
    {
        std::cerr << "Can't find any cloud points." << std::endl;
        std::cerr << "谢谢你" <<  std::endl;
    }
    else
    {   
        // run the viewer loop
        while (!vis->wasStopped()) vis->spinOnce(16);
    }

}
