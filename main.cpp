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

void extractAndSetClassifications(pdal::PointViewPtr classifiedView, pdal::PointViewPtr unclassifiedView) {
    const auto size = classifiedView->size();

    // Loop over each point in the classifiedView and copy its classification to the unclassifiedView
    for (size_t idx = 0; idx < size; ++idx) {
        uint8_t classification = classifiedView->getFieldAs<uint8_t>(pdal::Dimension::Id::Classification, idx);
        unclassifiedView->setField<uint8_t>(pdal::Dimension::Id::Classification, idx, classification);
    }
}

void processPointClouds(const fs::path& classifiedFile, const fs::path& unclassifiedFile, 
                       const fs::path& outputPath) {
    // Define LAS file path
    const std::string classifiedFilename = classifiedFile.string();
    const std::string unclassifiedFilename = classifiedFile.string();
    
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

    // pdal::Dimension::IdList dims = point_view->dims();
    // pdal::LasHeader las_header = las_reader.header();
    // int point_class = point_view->getFieldAs<int>(Id::Classification, idx);
    auto classifiedView = *classifiedPointViewSet.begin();
    auto unclassifiedView = *unclassPointViewSet.begin();

    auto size = classifiedView->size();
    for (size_t idx = 0; idx < size; ++idx) {
        std::cout << classifiedView->getFieldAs<int>(pdal::Dimension::Id::Classification, idx) << " ";
    }

    // Extract classifications from the 'Classification' dimension
    extractAndSetClassifications(classifiedView, unclassifiedView);
    // Use the below line somehow for the unclassified view
    //view->setField<uint8_t>(pdal::Dimension::Id::Classification,idx)
    // Iterate over views
    // Todo: Apply the classifications to the unclassified file
    for (size_t idx = 0; idx < size; ++idx) {
        std::cout << unclassifiedView->getFieldAs<int>(pdal::Dimension::Id::Classification, idx) << " ";
    }

    // Create a reader to take in the changed view
    pdal::BufferReader newViewReader;
    newViewReader.addView(unclassifiedView);

    pdal::Stage *writer = factory.createStage("writers.las");
    pdal::Options outputOptions;
    outputOptions.add("filename",outputPath.string());

    writer->setInput(newViewReader);
    writer->setOptions(outputOptions);
    writer->prepare(unclassifiedTable); // Can re-use target table as other properties have not changed
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