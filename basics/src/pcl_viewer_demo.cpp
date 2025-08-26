#include "viewer.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "Usage: pcl_viewer <cloud.pcd|.ply>\n"; return 1; }
  auto cloud = basics::loadCloud(argv[1]);
  auto vis   = basics::makeViewer(cloud, "Viewer demo");
  while (!vis->wasStopped()) vis->spinOnce(16);
  return 0;
}