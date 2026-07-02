#include "compound-config/compound-config.hpp"
#include "model/topology.hpp"
#include "model/buffer.hpp"
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    config::CompoundConfig config(argv[1]);
    auto root = config.getRoot();
    
    bool is_sparse = false;
    model::Topology::Specs specs = model::Topology::ParseTreeSpecs(root.lookup("architecture"), is_sparse);
    
    for (unsigned i = 0; i < specs.NumStorageLevels(); i++) {
        std::shared_ptr<model::LevelSpecs> level = specs.GetStorageLevel(i);
        std::cout << "Storage Level: " << level->level_name << std::endl;
        if (level->Type() == "BufferLevel") {
            auto buffer = std::static_pointer_cast<model::BufferLevel::Specs>(level);
            std::cout << "  Instances: " << buffer->instances.Get() << std::endl;
            std::cout << "  Size: " << buffer->size.Get() << std::endl;
            std::cout << "  Word bits: " << buffer->word_bits.Get() << std::endl;
        }
    }
    
    return 0;
}
