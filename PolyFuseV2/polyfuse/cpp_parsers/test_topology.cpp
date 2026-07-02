#include "compound-config/compound-config.hpp"
#include "model/topology.hpp"
#include "model/buffer.hpp"
#include "model/arithmetic.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    config::CompoundConfig config(argv[1]);
    auto root = config.getRoot();
    
    // Topology::ParseTreeSpecs needs the "architecture" node
    bool is_sparse = false;
    model::Topology::Specs specs = model::Topology::ParseTreeSpecs(root.lookup("architecture"), is_sparse);
    
    std::cout << "Levels parsed: " << specs.NumStorageLevels() << std::endl;
    for (unsigned i = 0; i < specs.NumStorageLevels(); i++) {
        std::shared_ptr<model::LevelSpecs> level = specs.GetStorageLevel(i);
        std::cout << "Storage Level: " << level->level_name << std::endl;
    }
    
    return 0;
}
