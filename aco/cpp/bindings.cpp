// Python module aco_cpp: the C++ BABC and MMAS used by aco/run.py --backend cpp, plus ratio and
// S-GROA for checking the port against aco/sgroa.py.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <optional>
#include "instance.hpp"
#include "babc.hpp"
#include "mmas.hpp"

namespace py = pybind11;

namespace {

using BoolArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

BoolArray to_bool_array(const std::vector<uint8_t>& v) {
    BoolArray out((py::ssize_t)v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::dict to_dict(const aco::Result& r) {
    py::dict d;
    d["solution"] = to_bool_array(r.solution);
    d["profit"] = r.profit;
    d["history"] = py::array_t<long long>((py::ssize_t)r.history.size(), r.history.data());
    d["evaluations"] = r.evaluations;
    d["elapsed"] = r.elapsed_sec;
    return d;
}

}  // namespace

PYBIND11_MODULE(aco_cpp, m) {
    m.doc() = "C++ BABC (He et al. 2018) and MAX-MIN Ant System for the SUKP";

    m.def("babc", [](const std::string& path, int n_bees, std::optional<int> max_iter, double a,
                     std::optional<double> limit, uint64_t seed) {
        Instance inst = load_instance(path);
        aco::BABCParams p;
        p.n_bees = n_bees;
        p.max_iter = max_iter.value_or(0);
        p.a = a;
        p.limit = limit.value_or(0);
        p.seed = seed;
        return to_dict(aco::babc(inst, p));
    }, py::arg("path"), py::arg("n_bees") = 20, py::arg("max_iter") = py::none(), py::arg("a") = 5.0,
       py::arg("limit") = py::none(), py::arg("seed") = 0,
       "Same parameters and result dict as aco.babc.babc (elapsed in seconds added); "
       "max_iter and limit default to max(m, n) and max(m, n) / 5");

    m.def("mmas", [](const std::string& path, int n_ants, std::optional<int> max_iter, double alpha, double beta,
                     double rho, double tau_min, const std::string& heuristic, uint64_t seed) {
        if (heuristic != "static" && heuristic != "dynamic")
            throw std::invalid_argument("heuristic must be 'static' or 'dynamic'");
        Instance inst = load_instance(path);
        aco::MMASParams p;
        p.n_ants = n_ants;
        p.max_iter = max_iter.value_or(0);
        p.alpha = alpha;
        p.beta = beta;
        p.rho = rho;
        p.tau_min = tau_min;
        p.dynamic = heuristic == "dynamic";
        p.seed = seed;
        return to_dict(aco::mmas(inst, p));
    }, py::arg("path"), py::arg("n_ants") = 20, py::arg("max_iter") = py::none(), py::arg("alpha") = 1.0,
       py::arg("beta") = 2.0, py::arg("rho") = 0.1, py::arg("tau_min") = 0.01, py::arg("heuristic") = "static",
       py::arg("seed") = 0, "Same parameters and result dict as aco.mmas.mmas (elapsed in seconds added)");

    m.def("ratio", [](const std::string& path) {
        Instance inst = load_instance(path);
        aco::Problem pb(inst);
        std::vector<double> r = aco::ratio(pb);
        return py::array_t<double>((py::ssize_t)r.size(), r.data());
    }, py::arg("path"), "p_i / R_i of every item, as aco.sgroa.ratio");

    m.def("sgroa", [](const std::string& path, BoolArray y) {
        Instance inst = load_instance(path);
        if (y.ndim() != 1 || y.size() != inst.m) throw std::invalid_argument("y must have one entry per item");
        aco::Problem pb(inst);
        aco::Cover cover(pb);
        std::vector<uint8_t> yv(y.data(), y.data() + y.size());
        aco::sgroa(cover, aco::ratio_order(pb), yv);
        return py::make_tuple(to_bool_array(cover.selected), cover.profit);
    }, py::arg("path"), py::arg("y"), "S-GROA of the bool mask y: (selected, profit), as aco.sgroa.sgroa");
}
