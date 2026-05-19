#include "kulik_a_mat_mul_double_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include "kulik_a_mat_mul_double_ccs/common/include/common.hpp"

namespace kulik_a_mat_mul_double_ccs {

namespace {

static void ProcessColumn(size_t j, const CCS &a, const CCS &b, double *accum, std::vector<double> &out_vals,
                          std::vector<size_t> &out_rows) {
  size_t row_min = std::numeric_limits<size_t>::max();
  size_t row_max = 0;

  for (size_t k = b.col_ind[j]; k < b.col_ind[j + 1]; ++k) {
    const size_t ind = b.row[k];
    const double b_val = b.value[k];
    for (size_t zc = a.col_ind[ind]; zc < a.col_ind[ind + 1]; ++zc) {
      const size_t i = a.row[zc];
      accum[i] += a.value[zc] * b_val;
      row_min = std::min(row_min, i);
      row_max = std::max(row_max, i);
    }
  }

  if (row_min <= row_max) {
    out_rows.reserve(row_max - row_min + 1);
    out_vals.reserve(row_max - row_min + 1);
    for (size_t i = row_min; i <= row_max; ++i) {
      if (accum[i] != 0.0) {
        out_rows.push_back(i);
        out_vals.push_back(accum[i]);
        accum[i] = 0.0;
      }
    }
  }
}

}  // namespace

KulikAMatMulDoubleCcsOMP::KulikAMatMulDoubleCcsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
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

  const int num_threads = omp_get_max_threads();
  std::vector<std::vector<double>> thread_accum(num_threads, std::vector<double>(a.n, 0.0));

#pragma omp parallel for default(none) schedule(static) shared(a, b, thread_accum, local_values, local_rows)
  for (size_t j = 0; j < b.m; ++j) {
    ProcessColumn(j, a, b, thread_accum[omp_get_thread_num()].data(), local_values[j], local_rows[j]);
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

#pragma omp parallel for default(none) schedule(static) shared(b, c, local_values, local_rows)
  for (size_t j = 0; j < b.m; ++j) {
    const size_t offset = c.col_ind[j];
    const size_t col_nz = local_values[j].size();
    const double *lv = local_values[j].data();
    const size_t *lr = local_rows[j].data();
    for (size_t k = 0; k < col_nz; ++k) {
      c.value[offset + k] = lv[k];
      c.row[offset + k] = lr[k];
    }
  }

  return true;
}

bool KulikAMatMulDoubleCcsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace kulik_a_mat_mul_double_ccs
