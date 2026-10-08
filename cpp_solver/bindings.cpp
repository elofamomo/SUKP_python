// Python module sukp_cpp: exposes the C++ solver to play.py and train_ranker.py.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "instance.hpp"
#include "solver.hpp"

namespace py = pybind11;

namespace {

template <typename T>
py::array_t<T> to_numpy(const std::vector<T>& v) {
    py::array_t<T> out((py::ssize_t)v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// Builds an instance from arrays (e.g. a generated training instance) instead of a file.
Instance make_instance(const std::vector<long long>& profits, const std::vector<long long>& weights,
                       const std::vector<std::vector<int>>& subsets, long long capacity) {
    if (subsets.size() != profits.size())
        throw std::invalid_argument("need one subset per item");
    Instance inst;
    inst.m = (int)profits.size();
    inst.n = (int)weights.size();
    inst.capacity = capacity;
    inst.profit = profits;
    inst.weight = weights;
    inst.subset_offset.assign(inst.m + 1, 0);
    for (int i = 0; i < inst.m; ++i) {
        for (int e : subsets[i]) {
            if (e < 0 || e >= inst.n) throw std::invalid_argument("element index out of range");
            inst.subset_flat.push_back(e);
        }
        inst.subset_offset[i + 1] = (int)inst.subset_flat.size();
    }
    return inst;
}

SUKPResult solve(const Instance& inst, const SUKPParams& params, py::object scorer) {
    SUKPSolver solver(inst, params);
    if (!scorer.is_none()) {
        solver.set_scorer([scorer](const std::vector<uint8_t>& selected, const std::vector<int>& candidates,
                                   std::vector<double>& scores) {
            py::object ret = scorer(to_numpy(selected), to_numpy(candidates));
            auto out = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(ret);
            if (!out || out.ndim() != 1 || (size_t)out.size() != candidates.size())
                throw std::runtime_error("scorer must return a 1-D array with one score per candidate");
            std::copy(out.data(), out.data() + out.size(), scores.begin());
        });
    }
    solver.set_interrupt_check([] { return PyErr_CheckSignals() != 0; });
    SUKPResult result = solver.run();
    if (PyErr_Occurred()) throw py::error_already_set();
    return result;
}

}  // namespace

PYBIND11_MODULE(sukp_cpp, m) {
    m.doc() = "C++ E2LS-style local search for the SUKP, with an optional learned add-back ranking";

    py::class_<Instance>(m, "Instance")
        .def_readonly("m", &Instance::m)
        .def_readonly("n", &Instance::n)
        .def_readonly("capacity", &Instance::capacity)
        .def_property_readonly("profits", [](const Instance& inst) { return to_numpy(inst.profit); })
        .def_property_readonly("weights", [](const Instance& inst) { return to_numpy(inst.weight); })
        .def_property_readonly("subsets", [](const Instance& inst) {
            py::list out;
            for (int i = 0; i < inst.m; ++i)
                out.append(to_numpy(std::vector<int>(inst.subset_flat.begin() + inst.subset_offset[i],
                                                     inst.subset_flat.begin() + inst.subset_offset[i + 1])));
            return out;
        });

    m.def("load_instance", &load_instance, py::arg("path"), "Load an instance in the SUKP_instances_60 format");
    m.def("make_instance", &make_instance, py::arg("profits"), py::arg("weights"), py::arg("subsets"),
          py::arg("capacity"), "Build an instance from profits, weights, per-item element lists and capacity");

    py::class_<SUKPParams>(m, "SolverParams")
        .def(py::init<>())
        .def_readwrite("rnum", &SUKPParams::rnum)
        .def_readwrite("anum", &SUKPParams::anum)
        .def_readwrite("construct_t", &SUKPParams::construct_t)
        .def_readwrite("add_back_node_budget", &SUKPParams::add_back_node_budget)
        .def_readwrite("time_budget_sec", &SUKPParams::time_budget_sec)
        .def_readwrite("max_iterations", &SUKPParams::max_iterations)
        .def_readwrite("seed", &SUKPParams::seed)
        .def_readwrite("sample_prob", &SUKPParams::sample_prob)
        .def_readwrite("max_samples", &SUKPParams::max_samples)
        .def_readwrite("oracle_node_budget", &SUKPParams::oracle_node_budget);

    py::class_<SUKPResult>(m, "SolverResult")
        .def_readonly("best_profit", &SUKPResult::best_profit)
        .def_readonly("elapsed_sec", &SUKPResult::elapsed_sec)
        .def_readonly("iterations", &SUKPResult::iterations)
        .def_readonly("restarts", &SUKPResult::restarts)
        .def_readonly("add_back_nodes", &SUKPResult::add_back_nodes)
        .def_readonly("scorer_calls", &SUKPResult::scorer_calls)
        .def_property_readonly("best_selected", [](const SUKPResult& r) { return to_numpy(r.best_selected); })
        .def_property_readonly("samples", [](const SUKPResult& r) {
            py::list out;
            for (const auto& s : r.samples)
                out.append(py::make_tuple(to_numpy(s.selected), to_numpy(s.candidates), to_numpy(s.values)));
            return out;
        }, "List of (selected uint8[m], candidates int32[k] in ratio order, values int64[k])");

    m.def("solve", &solve, py::arg("instance"), py::arg("params"), py::arg("scorer") = py::none(),
          "Run the local search. scorer(selected uint8[m], candidates int32[k]) -> float[k] replaces "
          "the value-weight ratio when ranking add-back candidates (higher is tried first).");
}
