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

    // Ground remover: plane ⟂ Z
    pcl::SACSegmentation<PointT> seg_ground;
    seg_ground.setOptimizeCoefficients(true);
    seg_ground.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg_ground.setMethodType(pcl::SAC_RANSAC);
    seg_ground.setAxis(Eigen::Vector3f(0,0,1));
    seg_ground.setEpsAngle(deg2rad(10.0f));
    seg_ground.setMaxIterations(500);
    seg_ground.setDistanceThreshold(0.05);

    // Cylinder RANSAC (with normals)
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg_cyl;
    seg_cyl.setOptimizeCoefficients(true);
    seg_cyl.setModelType(pcl::SACMODEL_CYLINDER);
    seg_cyl.setMethodType(pcl::SAC_RANSAC);
    seg_cyl.setAxis(Eigen::Vector3f(0,0,1));
    seg_cyl.setEpsAngle(deg2rad(10.0f));
    seg_cyl.setNormalDistanceWeight(0.15);
    seg_cyl.setMaxIterations(3000);
    seg_cyl.setDistanceThreshold(0.025);
    seg_cyl.setRadiusLimits(0.07, 0.30);

    // ---- Datasets ----
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_voxel(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_band(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_groundless(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_band(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_groundless(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coeff_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

    // ---- Load ----
    std::cout << "Reading: " << in_path << "\n";
    if (reader.read(in_path.string(), *cloud) < 0) { PCL_ERROR("Couldn't read file\n"); return -1; }
    std::cerr << "Raw points: " << cloud->size() << "\n";

    // 1) Height band FIRST (on raw cloud)
    pcl::PointCloud<PointT>::Ptr cloud_band_raw(new pcl::PointCloud<PointT>);
    band.setInputCloud(cloud);
    band.setFilterFieldName("z");
    band.setFilterLimits(0.3, 3.0);            // widen for now
    band.filter(*cloud_band_raw);

    // 2) THEN voxelize the already-cropped cloud
    vg.setInputCloud(cloud_band_raw);
    vg.setLeafSize(0.03f, 0.03f, 0.03f);       // 3 cm
    vg.filter(*cloud_voxel);

    // ---- De-spike ----
    ror.setInputCloud(cloud_band);
    ror.setRadiusSearch(0.08f);
    ror.setMinNeighborsInRadius(4);
    ror.filter(*cloud_band);

    // ---- Normals ----
    ne.setSearchMethod(kdtree);
    ne.setInputCloud(cloud_band);
    ne.setRadiusSearch(0.05f);
    ne.setViewPoint(0,0,10);
    ne.compute(*normals_band);

    // ---- Ground removal ----
    seg_ground.setInputCloud(cloud_band);
    seg_ground.segment(*inliers_plane, *coeff_plane);
    std::cerr << "Ground plane: " << *coeff_plane << "\n";

    ex_pts.setInputCloud(cloud_band);
    ex_pts.setIndices(inliers_plane);
    ex_pts.setNegative(true);
    ex_pts.filter(*cloud_groundless);

    ex_nrm.setInputCloud(normals_band);
    ex_nrm.setIndices(inliers_plane);
    ex_nrm.setNegative(true);
    ex_nrm.filter(*normals_groundless);

    // ---- Keep near-horizontal normals (optional) ----
    {
        pcl::PointIndices::Ptr horiz(new pcl::PointIndices);
        horiz->indices.reserve(normals_groundless->size());
        for (int i=0;i<(int)normals_groundless->size();++i){
            float nz = std::fabs(normals_groundless->points[i].normal_z);
            if (nz < 0.30f) horiz->indices.push_back(i);
        }
        pcl::PointCloud<PointT>::Ptr tmp_pts(new pcl::PointCloud<PointT>());
        pcl::PointCloud<pcl::Normal>::Ptr tmp_nrm(new pcl::PointCloud<pcl::Normal>());
        ex_pts.setInputCloud(cloud_groundless); ex_pts.setIndices(horiz); ex_pts.setNegative(false); ex_pts.filter(*tmp_pts);
        ex_nrm.setInputCloud(normals_groundless); ex_nrm.setIndices(horiz); ex_nrm.setNegative(false); ex_nrm.filter(*tmp_nrm);
        cloud_groundless.swap(tmp_pts); normals_groundless.swap(tmp_nrm);
    }

    // ---- Viewer ----
    auto vis = basics::makeViewer(cloud_groundless, "RANSAC — scene (groundless)");

    // ---- Iterative trunk extraction ----
    const int max_trees = 8;
    const std::size_t MIN_INLIERS = 10;

    pcl::PointCloud<PointT>::Ptr work_pts(new pcl::PointCloud<PointT>(*cloud_groundless));
    pcl::PointCloud<pcl::Normal>::Ptr work_nrm(new pcl::PointCloud<pcl::Normal>(*normals_groundless));

    while (!vis->wasStopped() ) {  // viewer keeps pumping
        if ((int)work_pts->size() == 0 || (int)work_nrm->size() == 0) { std::cerr << "No more data.\n"; break; }
        if (vis->getShapeActorMap()->size() / 2 >= (size_t)max_trees) break; // rough stop

        seg_cyl.setInputCloud(work_pts);
        seg_cyl.setInputNormals(work_nrm);

        pcl::ModelCoefficients::Ptr coeff_cyl(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers_cyl(new pcl::PointIndices);
        seg_cyl.segment(*inliers_cyl, *coeff_cyl);
        if (inliers_cyl->indices.empty()) { std::cerr << "No cylinder found.\n"; break; }

        // Extract cylinder points
        pcl::PointCloud<PointT>::Ptr cyl_pts(new pcl::PointCloud<PointT>());
        ex_pts.setInputCloud(work_pts);
        ex_pts.setIndices(inliers_cyl);
        ex_pts.setNegative(false);
        ex_pts.filter(*cyl_pts);

        // Helper to remove current inliers and recompute normals
        auto remove_current_and_continue = [&](){
            pcl::PointCloud<PointT>::Ptr tmp_pts(new pcl::PointCloud<PointT>());
            ex_pts.setNegative(true); ex_pts.filter(*tmp_pts);
            work_pts.swap(tmp_pts);

            pcl::PointCloud<pcl::Normal>::Ptr tmp_nrm(new pcl::PointCloud<pcl::Normal>());
            ex_nrm.setInputCloud(work_nrm); ex_nrm.setIndices(inliers_cyl);
            ex_nrm.setNegative(true); ex_nrm.filter(*tmp_nrm);
            work_nrm.swap(tmp_nrm);

            if (!work_pts->empty()) {
                ne.setInputCloud(work_pts);
                ne.setSearchMethod(kdtree);
                ne.setRadiusSearch(0.08f);
                ne.compute(*work_nrm);
            }
        };

        // Gates
        if (cyl_pts->size() < MIN_INLIERS) {
            std::cerr << "Reject: too few inliers (" << cyl_pts->size() << ")\n";
            remove_current_and_continue();
            continue;
        }

        Eigen::Vector3f axis(coeff_cyl->values[3], coeff_cyl->values[4], coeff_cyl->values[5]);
        axis.normalize();
        if (std::fabs(axis.dot(Eigen::Vector3f(0,0,1))) < 0.95f) {
            std::cerr << "Reject: axis not vertical enough\n";
            remove_current_and_continue();
            continue;
        }

        float zmin = +1e9f, zmax = -1e9f;
        for (int idx : inliers_cyl->indices) {
            float z = work_pts->points[idx].z;
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        if (zmax - zmin < 0.6f) {
            std::cerr << "Reject: vertical span too small\n";
            remove_current_and_continue();
            continue;
        }

        // Accept trunk
        Eigen::Vector4f c; pcl::compute3DCentroid(*cyl_pts, c);
        std::cout << "[Trunk] centroid: " << c[0] << ", " << c[1] << ", " << c[2]
                  << " | radius ≈ " << coeff_cyl->values[6] << " m\n";

        const auto out_i = out_dir / ("cylinder_" + std::to_string(vis->getShapeActorMap()->size()) + ".pcd");
        writer.write(out_i.string(), *cyl_pts, false);
        pcl::visualization::PointCloudColorHandlerCustom<PointT> color(cyl_pts, 20, 200, 20);
        const std::string cid = "trunk_" + std::to_string(vis->getShapeActorMap()->size());
        vis->addPointCloud<PointT>(cyl_pts, color, cid);
        vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cid);

        // Axis line
        Eigen::Vector3f p(coeff_cyl->values[0], coeff_cyl->values[1], coeff_cyl->values[2]);
        Eigen::Vector3f p1 = p + 5.0f*axis, p2 = p - 5.0f*axis;
        vis->addLine(pcl::PointXYZ(p1.x(),p1.y(),p1.z()),
                     pcl::PointXYZ(p2.x(),p2.y(),p2.z()),
                     1.0,0.5,0.1, "axis_"+std::to_string(vis->getShapeActorMap()->size()));

        remove_current_and_continue();
    }

    while (!vis->wasStopped()) vis->spinOnce(16);
    return 0;
}
