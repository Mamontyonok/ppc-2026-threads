#include "kulik_a_mat_mul_double_ccs/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <execution>
#include <numeric>
#include <tuple>
#include <vector>

#include "kulik_a_mat_mul_double_ccs/common/include/common.hpp"

namespace kulik_a_mat_mul_double_ccs {

namespace {

inline void ProcessColumn(size_t j, const CCS &a, const CCS &b, std::vector<double> &accum,
                          std::vector<bool> &nz_elem_rows, std::vector<size_t> &nnz_rows,
                          std::vector<std::vector<double>> &local_values,
                          std::vector<std::vector<size_t>> &local_rows) {
  for (size_t k = b.col_ind[j]; k < b.col_ind[j + 1]; ++k) {
    size_t ind = b.row[k];
    double b_val = b.value[k];
    for (size_t zc = a.col_ind[ind]; zc < a.col_ind[ind + 1]; ++zc) {
      size_t i = a.row[zc];
      double a_val = a.value[zc];

      accum[i] += a_val * b_val;
      if (!nz_elem_rows[i]) {
        nz_elem_rows[i] = true;
        nnz_rows.push_back(i);
      }
    }
  }

  std::ranges::sort(nnz_rows);

  for (size_t i : nnz_rows) {
    if (accum[i] != 0.0) {
      local_rows[j].push_back(i);
      local_values[j].push_back(accum[i]);
    }
    accum[i] = 0.0;
    nz_elem_rows[i] = false;
  }
  nnz_rows.clear();
}

inline void CopyColumn(size_t j, CCS &c, const std::vector<std::vector<double>> &local_values,
                       const std::vector<std::vector<size_t>> &local_rows) {
  size_t offset = c.col_ind[j];
  size_t col_nz = local_values[j].size();
  for (size_t k = 0; k < col_nz; ++k) {
    c.value[offset + k] = local_values[j][k];
    c.row[offset + k] = local_rows[j][k];
  }
}

}  // namespace

KulikAMatMulDoubleCcsSTL::KulikAMatMulDoubleCcsSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KulikAMatMulDoubleCcsSTL::ValidationImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  return (a.m == b.n);
}

bool KulikAMatMulDoubleCcsSTL::PreProcessingImpl() {
  return true;
}

bool KulikAMatMulDoubleCcsSTL::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  OutType &c = GetOutput();

  c.n = a.n;
  c.m = b.m;
  c.col_ind.assign(c.m + 1, 0);

  std::vector<std::vector<double>> local_values(b.m);
  std::vector<std::vector<size_t>> local_rows(b.m);

  std::vector<size_t> cols(b.m);
  std::iota(cols.begin(), cols.end(), 0);

  std::for_each(std::execution::par, cols.begin(), cols.end(), [&](size_t j) {
    thread_local std::vector<double> accum;
    thread_local std::vector<bool> nz_elem_rows;
    thread_local std::vector<size_t> nnz_rows;

    if (accum.size() < a.n) {
      accum.resize(a.n, 0.0);
      nz_elem_rows.resize(a.n, false);
    }

    ProcessColumn(j, a, b, accum, nz_elem_rows, nnz_rows, local_values, local_rows);
  });

  size_t total_nz = 0;
  for (size_t j = 0; j < b.m; ++j) {
    c.col_ind[j] = total_nz;
    total_nz += local_values[j].size();
  }
  c.col_ind[b.m] = total_nz;
  c.nz = total_nz;

  c.value.resize(total_nz);
  c.row.resize(total_nz);

  std::for_each(std::execution::par, cols.begin(), cols.end(), [&](size_t j) {
    CopyColumn(j, c, local_values, local_rows);
  });

  return true;
}

bool KulikAMatMulDoubleCcsSTL::PostProcessingImpl() {
  return true;
}

}  // namespace kulik_a_mat_mul_double_ccs
