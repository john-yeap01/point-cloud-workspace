#include <iostream>
#include <filesystem>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/io.h>          // for getFieldsList (optional but safe)
#include <pcl/io/pcd_io.h>


static const std::filesystem::path dataDir = DATA_DIR;  // must be a quoted macro
static const std::string filename = "forest3.pcd";

int main()
{
    const std::filesystem::path in_path  = dataDir / filename;
    const std::filesystem::path out_dir  = dataDir / "cloud_out";
    const std::filesystem::path out_path = out_dir / "forest3_no_ground.pcd";
    std::filesystem::create_directories(out_dir); // ensure cloud_out exists

    pcl::PCDReader reader; 
    pcl::PCDWriter writer;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    

    // Read the file
    if (reader.read(in_path.string(), *cloud) <0){
        std::cerr << "Failed to read " << in_path << "\n";
        return 1;
    }

    std::cerr << "Before: " << cloud->width * cloud->height
              << " points (" << pcl::getFieldsList(*cloud) << ")\n";

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 50.0);
    pass.filter(*cloud_filtered);

    std::cerr << "Cloud after filtering: " << std::endl;
    std::cerr << cloud->width * cloud->height << std::endl;


    float minZ = std::numeric_limits<float>::infinity();
    for (const pcl::PointXYZ& point: cloud->points){
        if (point.z < minZ){
            minZ = point.z;
        }
    }
    std::cout << "Minimum z of all point in the cloud: " << minZ << std::endl;


    if (writer.write(out_path.string(), *cloud_filtered, false)) {
        std::cerr << "Failed to write " << out_path << "\n";
        return 1;
    }
    std::cout << "Wrote " << out_path << "\n";
    return (0);
}
