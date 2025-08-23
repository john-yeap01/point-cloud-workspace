#include <iostream>
#include <filesystem>

#include <pcl/PCLPointCloud2.h>
#include <pcl/common/io.h>          // for getFieldsList (optional but safe)
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

static const std::filesystem::path dataDir = DATA_DIR;  // must be a quoted macro
static const std::string filename = "table_scene_lms400.pcd";

int main() {
    const std::filesystem::path in_path  = dataDir / filename;
    const std::filesystem::path out_dir  = dataDir / "cloud_out";
    const std::filesystem::path out_path = out_dir / "table_scene_lms400_downsampled.pcd";
    std::filesystem::create_directories(out_dir); // ensure cloud_out exists

    std::cout << "Running voxel downsampling\n";
    std::cout << "Opening " << in_path << "\n";

    // clouds
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2);

    // read
    pcl::PCDReader reader;
    if (reader.read(in_path.string(), *cloud) < 0) {
        std::cerr << "Failed to read " << in_path << "\n";
        return 1;
    }

    std::cerr << "Before: " << cloud->width * cloud->height
              << " points (" << pcl::getFieldsList(*cloud) << ")\n";

    // filter
    pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);   // 1 cm voxels
    vg.filter(*cloud_filtered);            

    std::cerr << "After : " << cloud_filtered->width * cloud_filtered->height
              << " points (" << pcl::getFieldsList(*cloud_filtered) << ")\n";

    // write
    pcl::PCDWriter writer;
    if (writer.write(out_path.string(), *cloud_filtered,
                     Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false) < 0) {
        std::cerr << "Failed to write " << out_path << "\n";
        return 1;
    }
    std::cout << "Wrote " << out_path << "\n";
    return 0;
}
