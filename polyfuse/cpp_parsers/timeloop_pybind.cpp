#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "workload/workload.hpp"
#include "compound-config/compound-config.hpp"
#include "model/topology.hpp"
#include "model/buffer.hpp"
#include <iostream>
#include <limits>

namespace py = pybind11;

std::map<std::string, int> parse_problem(const std::string& path) {
    config::CompoundConfig config(path.c_str());
    problem::Workload workload;
    
    // Parse the workload instance using Timeloop's API
    problem::ParseWorkload(config.getRoot().lookup("problem"), workload);
    
    std::map<std::string, int> dimensions;
    const problem::Shape* shape = problem::GetShape();
    
    // Factorized bounds (C, M, R, S, P, Q, N, etc)
    for (unsigned i = 0; i < shape->NumFactorizedDimensions; i++) {
        std::string name = shape->FactorizedDimensionIDToName.at(i);
        dimensions[name] = workload.GetFactorizedBound(i);
    }
    
    // Coefficients (Wstride, Hstride, etc)
    for (unsigned i = 0; i < shape->NumCoefficients; i++) {
        std::string name = shape->CoefficientIDToName.at(i);
        dimensions[name] = workload.GetCoefficient(i);
    }
    
    return dimensions;
}

py::dict parse_architecture(const std::string& arch_path, const std::string& ert_path = "", const std::string& art_path = "") {
    config::CompoundConfig config(arch_path.c_str());
    auto root = config.getRoot();
    
    bool is_sparse = false;
    model::Topology::Specs specs = model::Topology::ParseTreeSpecs(root.lookup("architecture"), is_sparse);
    
    // Native ERT parsing via Timeloop
    if (!ert_path.empty()) {
        config::CompoundConfig ert_config(ert_path.c_str());
        specs.ParseAccelergyERT(ert_config.getRoot().lookup("ERT"));
    }
    
    // Native ART parsing via Timeloop
    if (!art_path.empty()) {
        config::CompoundConfig art_config(art_path.c_str());
        specs.ParseAccelergyART(art_config.getRoot().lookup("ART"));
    }
    
    py::dict hw_config;
    py::dict buffers;
    py::list memory_hierarchy;
    
    // Iterate through storage levels (innermost to outermost)
    for (unsigned i = 0; i < specs.NumStorageLevels(); i++) {
        std::shared_ptr<model::LevelSpecs> level = specs.GetStorageLevel(i);
        std::string name = level->level_name;
        
        if (level->Type() == "BufferLevel") {
            auto buffer = std::static_pointer_cast<model::BufferLevel::Specs>(level);
            py::dict buf_dict;
            buf_dict["name"] = name;
            buf_dict["tree_depth"] = i;
            
            buf_dict["instances"] = buffer->instances.IsSpecified() ? buffer->instances.Get() : 1;
            
            if (buffer->size.IsSpecified()) {
                buf_dict["capacity_words"] = buffer->size.Get();
                buf_dict["total_capacity_words"] = buffer->size.Get() * buf_dict["instances"].cast<int>();
            } else {
                buf_dict["capacity_words"] = std::numeric_limits<double>::infinity();
                buf_dict["total_capacity_words"] = std::numeric_limits<double>::infinity();
            }
            
            buf_dict["word_bits"] = buffer->word_bits.IsSpecified() ? buffer->word_bits.Get() : 16;
            buf_dict["datawidth"] = buf_dict["word_bits"];
            buf_dict["width"] = buf_dict["word_bits"];
            
            buf_dict["cluster_size"] = buffer->cluster_size.IsSpecified() ? buffer->cluster_size.Get() : 1;
            
            // Extract the ERT resolved energy (if ERT was parsed)
            double energy = 0.0;
            if (buffer->op_energy_map.find("read") != buffer->op_energy_map.end()) {
                energy = buffer->op_energy_map["read"];
            } else if (buffer->vector_access_energy.IsSpecified()) {
                energy = buffer->vector_access_energy.Get();
            }
            buf_dict["energy_per_access"] = energy;
            
            buffers[name.c_str()] = buf_dict;
            memory_hierarchy.append(name);
        }
    }
    
    // Extract Arithmetic/MAC energy
    auto mac_level = specs.GetArithmeticLevel();
    double mac_energy = 0.0;
    if (mac_level) {
        if (mac_level->op_energy_map.find("mac") != mac_level->op_energy_map.end()) {
            mac_energy = mac_level->op_energy_map["mac"];
        } else if (mac_level->op_energy_map.find("compute") != mac_level->op_energy_map.end()) {
            mac_energy = mac_level->op_energy_map["compute"];
        } else if (mac_level->energy_per_op.IsSpecified()) {
            mac_energy = mac_level->energy_per_op.Get();
        }
    }
    hw_config["mac_energy"] = mac_energy;
    
    hw_config["buffers"] = buffers;
    hw_config["memory_hierarchy"] = memory_hierarchy;
    
    return hw_config;
}

PYBIND11_MODULE(timeloop_pybind, m) {
    m.def("parse_problem", &parse_problem, "Parse Timeloop problem YAML");
    m.def("parse_architecture", &parse_architecture, "Parse Timeloop architecture YAML",
          py::arg("arch_path"), py::arg("ert_path") = "", py::arg("art_path") = "");
}
