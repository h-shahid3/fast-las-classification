#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Stage.hpp>
#include <pdal/Dimension.hpp>
#include <algorithm>
#include <execution>
#include <ranges>
#include <numeric>
#include <io/BufferReader.hpp>
#include <nanoflann.hpp>

namespace fs = std::filesystem;

std::vector<fs::path> getInputFiles(const std::string& input) {
    std::vector<fs::path> files;
    
    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (entry.path().extension() == ".las" || entry.path().extension() == ".laz") {
                files.push_back(entry.path());
            }
        }
    } else {
        // Assume it's a text file containing paths. Currently lacks extensions checking for each file, which should be implemented.
        std::ifstream inFile(input);
        std::string line;
        while (std::getline(inFile, line)) {
            if (!line.empty()) {
                files.push_back(line);
            }
        }
    }
    return files;
}

std::map<std::string, fs::path> createFileMap(const std::vector<fs::path>& files) {
    std::map<std::string, fs::path> fileMap;
    for (const auto& file : files) {
        fileMap[file.stem().string()] = file;
    }
    return fileMap;
}


void printUsage() {
    std::cout << "Usage: point_cloud_classifier <classified> <unclassified> [output_dir]\n"
              << "  classified: Directory containing classified point clouds or text file with paths\n"
              << "  unclassified: Directory containing unclassified point clouds or text file with paths\n"
              << "  output_dir: Optional. Directory for output files. If not specified, unclassified files "
              << "will be overwritten with the copied classifications\n";
}

// Point cloud adapter for nanoflann - simplified for k=1 case
struct PointCloud {
    struct Point {
        double x, y, z;
        uint8_t classification;
        
        Point(double x_, double y_, double z_, uint8_t c = 0) 
            : x(x_), y(y_), z(z_), classification(c) {}
    };
    
    std::vector<Point> points;
    
    inline size_t kdtree_get_point_count() const { return points.size(); }
    
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return points[idx].x;
        if (dim == 1) return points[idx].y;
        return points[idx].z;
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    3
>;

void extractAndSetClassifications(pdal::PointViewPtr classifiedView, pdal::PointViewPtr unclassifiedView) {
    // Create point cloud for classified points
    PointCloud classifiedCloud;
    classifiedCloud.points.reserve(classifiedView->size());
    
    // Fill the classified point cloud
    for (size_t idx = 0; idx < classifiedView->size(); ++idx) {
        classifiedCloud.points.emplace_back(
            classifiedView->getFieldAs<double>(pdal::Dimension::Id::X, idx),
            classifiedView->getFieldAs<double>(pdal::Dimension::Id::Y, idx),
            classifiedView->getFieldAs<double>(pdal::Dimension::Id::Z, idx),
            classifiedView->getFieldAs<uint8_t>(pdal::Dimension::Id::Classification, idx)
        );
    }
    
    // Construct a kd-tree index
    KDTree index(3, classifiedCloud, {10 /* max leaf size */});
    index.buildIndex();
    
    // Process unclassified points in parallel
    const size_t num_points = unclassifiedView->size();
    std::vector<uint8_t> new_classifications(num_points);
    
    #pragma omp parallel for
    for (size_t idx = 0; idx < num_points; ++idx) {
        // Get point coordinates
        double query_pt[3] = {
            unclassifiedView->getFieldAs<double>(pdal::Dimension::Id::X, idx),
            unclassifiedView->getFieldAs<double>(pdal::Dimension::Id::Y, idx),
            unclassifiedView->getFieldAs<double>(pdal::Dimension::Id::Z, idx)
        };
        
        // Single nearest neighbor search
        size_t ret_index;
        double out_dist_sqr;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        
        index.findNeighbors(resultSet, query_pt);
        
        // Directly use classification from nearest neighbor
        new_classifications[idx] = classifiedCloud.points[ret_index].classification;
    }
    
    // Apply new classifications
    for (size_t idx = 0; idx < num_points; ++idx) {
        unclassifiedView->setField<uint8_t>(
            pdal::Dimension::Id::Classification, 
            idx, 
            new_classifications[idx]
        );
    }
}

void processPointClouds(const fs::path& classifiedFile, const fs::path& unclassifiedFile, 
                       const fs::path& outputPath) {
    // Define LAS file path
    const std::string classifiedFilename = classifiedFile.string();
    const std::string unclassifiedFilename = unclassifiedFile.string();
    
    // Create a classifiedReader
    pdal::StageFactory factory;
    pdal::Stage* classifiedReader = factory.createStage("readers.las");
    pdal::Stage* unclassifiedReader = factory.createStage("readers.las");

    // Set the filename as an option for the classifiedReader
    pdal::Options classOptions;
    classOptions.add("filename", classifiedFilename);
    classifiedReader->setOptions(classOptions);

    pdal::Options unclassOptions;
    unclassOptions.add("filename",unclassifiedFilename);
    unclassifiedReader->setOptions(unclassOptions);

    // Prepare PointTable to store metadata and data type
    pdal::PointTable classifiedTable;
    classifiedReader->prepare(classifiedTable);

    // Execute the pipeline and retrieve PointViews
    pdal::PointViewSet classifiedPointViewSet = classifiedReader->execute(classifiedTable);

    // Prepare PointTable to store metadata and data type
    pdal::PointTable unclassifiedTable;
    unclassifiedReader->prepare(unclassifiedTable);

    // Execute the pipeline and retrieve PointViews
    pdal::PointViewSet unclassPointViewSet = unclassifiedReader->execute(unclassifiedTable);

    auto classifiedView = *classifiedPointViewSet.begin(); // Assume there is only one PointView. Might need to review this
    auto unclassifiedView = *unclassPointViewSet.begin();

    // Extract classifications from the 'Classification' dimension
    extractAndSetClassifications(classifiedView, unclassifiedView);

    // Create a reader to take in the changed view
    pdal::BufferReader newViewReader;
    newViewReader.addView(unclassifiedView);

    pdal::Stage *writer = factory.createStage("writers.las");
    pdal::Options outputOptions;
    outputOptions.add("filename",outputPath.string());

    writer->setInput(newViewReader);
    writer->setOptions(outputOptions);
    writer->prepare(unclassifiedTable); // Should be able to re-use target table as other properties have not changed
    writer->execute(unclassifiedTable);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        printUsage();
        return 1;
    }

    std::string classifiedInput = argv[1];
    std::string unclassifiedInput = argv[2];
    std::string outputDir = (argc == 4) ? argv[3] : "";

    // Get classified and unclassified files
    auto classifiedFiles = getInputFiles(classifiedInput);
    auto unclassifiedFiles = getInputFiles(unclassifiedInput);

    // Create maps of filename stems to full paths
    auto classifiedMap = createFileMap(classifiedFiles);
    auto unclassifiedMap = createFileMap(unclassifiedFiles);

    // Process matching files
    for (const auto& [stem, classifiedPath] : classifiedMap) {
        auto unclassifiedIter = unclassifiedMap.find(stem);
        if (unclassifiedIter != unclassifiedMap.end()) {
            fs::path outputPath;
            if (outputDir.empty()) {
                outputPath = unclassifiedIter->second;
            } else {
                outputPath = fs::path(outputDir) / (stem + unclassifiedIter->second.extension().string());
            }

            std::cout << "Processing " << stem << "..." << std::endl;
            processPointClouds(classifiedPath, unclassifiedIter->second, outputPath);
        }
    }

    return 0;
}