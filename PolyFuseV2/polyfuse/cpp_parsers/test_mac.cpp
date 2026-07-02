#include "compound-config/compound-config.hpp"
#include "model/topology.hpp"
#include "model/arithmetic.hpp"
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    config::CompoundConfig config(argv[1]);
    auto root = config.getRoot();
    
    bool is_sparse = false;
    model::Topology::Specs specs = model::Topology::ParseTreeSpecs(root.lookup("architecture"), is_sparse);
    
    if (argc > 2) {
        config::CompoundConfig ert_config(argv[2]);
        specs.ParseAccelergyERT(ert_config.getRoot().lookup("ERT"));
    }
    
    auto mac_level = specs.GetArithmeticLevel();
    if (mac_level) {
        std::cout << "MAC op_energy_map size: " << mac_level->op_energy_map.size() << std::endl;
        for (auto const& [k, v] : mac_level->op_energy_map) {
            std::cout << k << ": " << v << std::endl;
        }
        if (mac_level->energy_per_op.IsSpecified()) {
            std::cout << "Energy per op: " << mac_level->energy_per_op.Get() << std::endl;
        }
    }
    return 0;
}
