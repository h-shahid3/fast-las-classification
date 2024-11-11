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
        // Assume it's a text file containing paths. Currently lacks extensions checking.
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

auto extractClassifications(std::shared_ptr<pdal::PointView> view) {
    const auto size = view->size();
    
    // Create vector of indices
    std::vector<size_t> indices(size);
    // Fill the vector with indices starting at 0 and ending at size - 1.
    std::iota(indices.begin(), indices.end(), 0);
    
    // Create output vector
    std::vector<uint8_t> classifications(size);
    
    // Transform indices to classifications in parallel
    std::transform(
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        classifications.begin(),
        [view](size_t idx) {
            return view->getFieldAs<uint8_t>(pdal::Dimension::Id::Classification, idx);
        }
    );
    
    return classifications;
}

void processPointClouds(const fs::path& classifiedFile, const fs::path& unclassifiedFile, 
                       const fs::path& outputPath) {
    // Define LAS file path
    const std::string filename = classifiedFile.string();
    
    // Create a reader
    pdal::StageFactory factory;
    pdal::Stage* reader = factory.createStage("readers.las");

    // Set the filename as an option for the reader
    pdal::Options options;
    options.add("filename", filename);
    reader->setOptions(options);

    // Prepare PointTable to store metadata and data type
    pdal::PointTable table;
    reader->prepare(table);

    // Execute the pipeline and retrieve PointViews
    pdal::PointViewSet pointViewSet = reader->execute(table);
    // Iterate over views
    for (auto view : pointViewSet) {
        // Extract classifications from the 'Classification' dimension
        std::vector<uint8_t> classifications = extractClassifications(view);

        // Todo: Apply the classifications to the unclassified file
        for (auto cls : classifications){
            std::cout << static_cast<int>(cls) << " ";
        }
    }
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