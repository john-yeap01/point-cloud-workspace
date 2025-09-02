// HEIGHT ABOVE GROUND 

#include <iostream>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <cstdint>
#include <cmath>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>     // getMinMax3D
#include <pcl/filters/filter.h>    // removeNaNFromPointCloud

static const std::filesystem::path dataDir = DATA_DIR;  // must be quoted macro in CMake
static const std::string filename = "forest3.pcd";

// ---- Tunables (start here) ----
static const float GRID_CELL   = 0.50f;  // meters (0.25–1.0 typical). Larger = faster, smoother.
static const float HAG_KEEP    = 0.35f;  // meters. Keep points with height-above-ground >= this.
static const bool  BINARY_IO   = true;   // faster PCD write
// --------------------------------

using PointT = pcl::PointXYZ;

// pack (i,j) into a single 64-bit key for unordered_map
inline std::uint64_t ij_key(int i, int j) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) << 32) ^
            static_cast<std::uint32_t>(j);
}

int main()
{
    const std::filesystem::path in_path   = dataDir / filename;
    const std::filesystem::path out_dir   = dataDir / "cloud_out";
    const std::filesystem::path out_nong  = out_dir / "forest3_no_ground.pcd";
    const std::filesystem::path out_ground= out_dir / "forest3_ground_only.pcd";
    std::filesystem::create_directories(out_dir); // ensure cloud_out exists

    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr clean (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr nong  (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr ground(new pcl::PointCloud<PointT>);

    // ── 1) Read
    if (reader.read(in_path.string(), *cloud) < 0) {
        std::cerr << "Failed to read " << in_path << "\n";
        return 1;
    }
    std::cerr << "Loaded: " << cloud->width * cloud->height
              << " points (" << pcl::getFieldsList(*cloud) << ")\n";

    // ── 2) Remove NaNs
    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*cloud, *clean, idx);
    std::cerr << "After NaN removal: " << clean->size() << " points\n";
    if (clean->empty()) {
        std::cerr << "Empty cloud after NaN removal.\n";
        return 1;
    }

    // ── 3) Compute bounds (for reference/logs)
    PointT minPt, maxPt;
    pcl::getMinMax3D(*clean, minPt, maxPt);
    std::cerr << "Z range: [" << minPt.z << ", " << maxPt.z << "]\n";

    // ── 4) Build min-Z (ground) raster on a coarse XY grid
    const float inv_cell = 1.0f / GRID_CELL;

    struct MinZ { float z; bool set; };
    std::unordered_map<std::uint64_t, MinZ> grid;
    grid.reserve(clean->size() / 16); // rough guess to reduce rehash

    for (const auto& p : clean->points) {
        int i = static_cast<int>(std::floor((p.x - minPt.x) * inv_cell));
        int j = static_cast<int>(std::floor((p.y - minPt.y) * inv_cell));
        auto key = ij_key(i, j);
        auto it = grid.find(key);
        if (it == grid.end()) {
            grid.emplace(key, MinZ{p.z, true});
        } else {
            if (p.z < it->second.z) it->second.z = p.z;
        }
    }
    std::cerr << "Grid cells with data: " << grid.size() << "\n";

    // (Optional) You can add a tiny 3x3 neighbor smoothing here if needed.
    // For speed, we skip it. If you see speckle ground, bump GRID_CELL to 0.7–1.0 m.

    // ── 5) Classify by HAG and split into ground / non-ground
    nong->points.reserve(clean->points.size());   // upper bound
    ground->points.reserve(clean->points.size());

    const float hag_keep = HAG_KEEP;

    for (const auto& p : clean->points) {
        int i = static_cast<int>(std::floor((p.x - minPt.x) * inv_cell));
        int j = static_cast<int>(std::floor((p.y - minPt.y) * inv_cell));
        auto it = grid.find(ij_key(i, j));

        // If this cell has no ground estimate (should be rare), be conservative: keep as non-ground.
        if (it == grid.end() || !it->second.set) {
            nong->points.push_back(p);
            continue;
        }

        float hag = p.z - it->second.z;
        if (hag >= hag_keep) {
            nong->points.push_back(p);
        } else {
            ground->points.push_back(p);
        }
    }

    nong->width  = static_cast<std::uint32_t>(nong->points.size());
    nong->height = 1;
    nong->is_dense = true;

    ground->width  = static_cast<std::uint32_t>(ground->points.size());
    ground->height = 1;
    ground->is_dense = true;

    std::cerr << "Ground pts:    " << ground->size() << "\n";
    std::cerr << "Non-ground pts:" << nong->size()   << "\n";

    // ── 6) Save (binary = faster)
    if (writer.write(out_nong.string(), *nong, /*binary=*/BINARY_IO) != 0) {
        std::cerr << "Failed to write " << out_nong << "\n";
        return 1;
    }
    if (writer.write(out_ground.string(), *ground, /*binary=*/BINARY_IO) != 0) {
        std::cerr << "Failed to write " << out_ground << "\n";
        return 1;
    }

    std::cout << "Wrote:\n  " << out_nong << "\n  " << out_ground << "\n";
    return 0;
}
