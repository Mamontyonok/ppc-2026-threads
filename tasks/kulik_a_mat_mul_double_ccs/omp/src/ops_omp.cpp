#include "kulik_a_mat_mul_double_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
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
    const size_t ind = b.row[k];
    const double b_val = b.value[k];
    for (size_t zc = a.col_ind[ind]; zc < a.col_ind[ind + 1]; ++zc) {
      const size_t i = a.row[zc];
      const double a_val = a.value[zc];

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
  const size_t offset = c.col_ind[j];
  const size_t col_nz = local_values[j].size();

  for (size_t k = 0; k < col_nz; ++k) {
    c.value[offset + k] = local_values[j][k];
    c.row[offset + k] = local_rows[j][k];
  }
}

void ProcessColumnsRange(size_t jstart, size_t jend, const CCS &a, const CCS &b,
                         std::vector<std::vector<double>> &local_values, std::vector<std::vector<size_t>> &local_rows) {
  std::vector<double> accum(a.n, 0.0);
  std::vector<bool> nz_elem_rows(a.n, false);
  std::vector<size_t> nnz_rows;
  nnz_rows.reserve(a.n);

  for (size_t j = jstart; j < jend; ++j) {
    ProcessColumn(j, a, b, accum, nz_elem_rows, nnz_rows, local_values, local_rows);
  }
}

}  // namespace

KulikAMatMulDoubleCcsOMP::KulikAMatMulDoubleCcsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCS();
}

bool KulikAMatMulDoubleCcsOMP::ValidationImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  return (a.m == b.n);
}

bool KulikAMatMulDoubleCcsOMP::PreProcessingImpl() {
  return true;
}

bool KulikAMatMulDoubleCcsOMP::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  OutType &c = GetOutput();

  c.n = a.n;
  c.m = b.m;
  c.col_ind.assign(c.m + 1, 0);

  std::vector<std::vector<double>> local_values(b.m);
  std::vector<std::vector<size_t>> local_rows(b.m);

  const int num_threads_raw = omp_get_max_threads();
  const int num_threads = std::max(1, num_threads_raw);
  const int threads_count = std::max(1, std::min(num_threads, static_cast<int>(b.m == 0 ? 1 : b.m)));

#pragma omp parallel default(none) shared(a, b, local_values, local_rows, threads_count)
  {
    const int tid = omp_get_thread_num();
    const size_t jstart = (static_cast<size_t>(tid) * b.m) / static_cast<size_t>(threads_count);
    const size_t jend = (static_cast<size_t>(tid + 1) * b.m) / static_cast<size_t>(threads_count);

    ProcessColumnsRange(jstart, jend, a, b, local_values, local_rows);
  }

  size_t total_nz = 0;
  for (size_t j = 0; j < b.m; ++j) {
    c.col_ind[j] = total_nz;
    total_nz += local_values[j].size();
  }
  c.col_ind[b.m] = total_nz;
  c.nz = total_nz;

  c.value.resize(total_nz);
  c.row.resize(total_nz);

#pragma omp parallel for default(none) schedule(static) shared(c, b, local_values, local_rows)
  for (size_t j = 0; j < b.m; ++j) {
    CopyColumn(j, c, local_values, local_rows);
  }

  return true;
}

bool KulikAMatMulDoubleCcsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace kulik_a_mat_mul_double_ccs
