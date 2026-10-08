#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Mirrors the SUKP_instances_60 text format parsed by helper/loader.py:
//   blank, blank, "m=<m> n=<n> knapsack size=<capacity>", blank,
//   "The profit of <m> items", <m profits>, blank,
//   "The weight of <n> elements", <n weights>, blank,
//   "Relation matrix", <m*n 0/1 entries row-major>
struct Instance {
    int m = 0, n = 0;
    long long capacity = 0;
    std::vector<long long> profit;        // size m
    std::vector<long long> weight;        // size n
    std::vector<int> subset_offset;       // size m+1 (CSR row offsets)
    std::vector<int> subset_flat;         // element indices per item, concatenated
};

inline long long parse_after_equals(const std::string& tok) {
    auto pos = tok.find('=');
    if (pos == std::string::npos) throw std::runtime_error("expected '=' in token: " + tok);
    return std::stoll(tok.substr(pos + 1));
}

inline std::string read_nonempty_line(std::ifstream& f) {
    std::string line;
    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        return line;
    }
    throw std::runtime_error("unexpected end of file while reading instance");
}

inline Instance load_instance(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open instance file: " + path);

    Instance inst;

    std::string header = read_nonempty_line(f);
    {
        std::istringstream iss(header);
        std::vector<std::string> toks;
        std::string t;
        while (iss >> t) toks.push_back(t);
        if (toks.size() < 4) throw std::runtime_error("malformed header line: " + header);
        inst.m = (int)parse_after_equals(toks[0]);
        inst.n = (int)parse_after_equals(toks[1]);
        inst.capacity = parse_after_equals(toks[3]);
    }

    std::string profit_header = read_nonempty_line(f);
    if (profit_header.rfind("The profit of", 0) != 0)
        throw std::runtime_error("expected profit header, got: " + profit_header);
    inst.profit.reserve(inst.m);
    while ((int)inst.profit.size() < inst.m) {
        std::istringstream iss(read_nonempty_line(f));
        long long v;
        while (iss >> v) inst.profit.push_back(v);
    }
    if ((int)inst.profit.size() != inst.m) throw std::runtime_error("profit count mismatch");

    std::string weight_header = read_nonempty_line(f);
    if (weight_header.rfind("The weight of", 0) != 0)
        throw std::runtime_error("expected weight header, got: " + weight_header);
    inst.weight.reserve(inst.n);
    while ((int)inst.weight.size() < inst.n) {
        std::istringstream iss(read_nonempty_line(f));
        long long v;
        while (iss >> v) inst.weight.push_back(v);
    }
    if ((int)inst.weight.size() != inst.n) throw std::runtime_error("weight count mismatch");

    std::string rel_header = read_nonempty_line(f);
    if (rel_header.find("Relation matrix") == std::string::npos)
        throw std::runtime_error("expected 'Relation matrix' header, got: " + rel_header);

    std::vector<std::vector<int>> subsets(inst.m);
    {
        std::vector<int> flat;
        flat.reserve((size_t)inst.m * (size_t)inst.n);
        int v;
        while (f >> v) flat.push_back(v);
        if ((long long)flat.size() != (long long)inst.m * (long long)inst.n)
            throw std::runtime_error("relation matrix size mismatch");
        for (int i = 0; i < inst.m; ++i)
            for (int j = 0; j < inst.n; ++j)
                if (flat[(size_t)i * inst.n + j]) subsets[i].push_back(j);
    }

    inst.subset_offset.resize(inst.m + 1, 0);
    for (int i = 0; i < inst.m; ++i)
        inst.subset_offset[i + 1] = inst.subset_offset[i] + (int)subsets[i].size();
    inst.subset_flat.reserve(inst.subset_offset[inst.m]);
    for (int i = 0; i < inst.m; ++i)
        for (int e : subsets[i]) inst.subset_flat.push_back(e);

    return inst;
}
