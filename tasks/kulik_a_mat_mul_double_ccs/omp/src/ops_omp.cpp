#include "kulik_a_mat_mul_double_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include "kulik_a_mat_mul_double_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kulik_a_mat_mul_double_ccs {

void KulikAMatMulDoubleCcsOMP::ProcessColumn(size_t j, int tid, const CCS &a, const CCS &b,
                                             std::vector<std::vector<double>> &thread_accum,
                                             std::vector<std::vector<bool>> &thread_nz,
                                             std::vector<std::vector<size_t>> &thread_nnz_rows,
                                             std::vector<std::vector<double>> &flat_vals,
                                             std::vector<std::vector<size_t>> &flat_rows,
                                             std::vector<size_t> &col_nnz) {
  for (size_t k = b.col_ind[j]; k < b.col_ind[j + 1]; ++k) {
    size_t ind = b.row[k];
    double b_val = b.value[k];
    for (size_t zc = a.col_ind[ind]; zc < a.col_ind[ind + 1]; ++zc) {
      size_t i = a.row[zc];
      thread_accum[tid][i] += a.value[zc] * b_val;
      if (!thread_nz[tid][i]) {
        thread_nz[tid][i] = true;
        thread_nnz_rows[tid].push_back(i);
      }
    }
  }

  std::ranges::sort(thread_nnz_rows[tid]);

  size_t cnt = 0;
  for (size_t i : thread_nnz_rows[tid]) {
    if (thread_accum[tid][i] != 0.0) {
      flat_rows[tid].push_back(i);
      flat_vals[tid].push_back(thread_accum[tid][i]);
      ++cnt;
    }
    thread_accum[tid][i] = 0.0;
    thread_nz[tid][i] = false;
  }
  thread_nnz_rows[tid].clear();
  col_nnz[j] = cnt;
}

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

  const int num_threads = ppc::util::GetNumThreads();

  std::vector<std::vector<double>> thread_accum(num_threads, std::vector<double>(a.n, 0.0));
  std::vector<std::vector<bool>> thread_nz(num_threads, std::vector<bool>(a.n, false));
  std::vector<std::vector<size_t>> thread_nnz_rows(num_threads);

  std::vector<std::vector<double>> flat_vals(num_threads);
  std::vector<std::vector<size_t>> flat_rows(num_threads);

  std::vector<size_t> col_nnz(b.m, 0);

#pragma omp parallel for default(none) schedule(static) \
    shared(a, b, thread_accum, thread_nz, thread_nnz_rows, flat_vals, flat_rows, col_nnz)
  for (size_t j = 0; j < b.m; ++j) {
    int tid = omp_get_thread_num();
    ProcessColumn(j, tid, a, b, thread_accum, thread_nz, thread_nnz_rows, flat_vals, flat_rows, col_nnz);
  }

  size_t total_nz = 0;
  for (size_t j = 0; j < b.m; ++j) {
    c.col_ind[j] = total_nz;
    total_nz += col_nnz[j];
  }
  c.col_ind[b.m] = total_nz;
  c.nz = total_nz;

  c.value.resize(total_nz);
  c.row.resize(total_nz);

#pragma omp parallel for default(none) schedule(static) shared(b, c, col_nnz, flat_vals, flat_rows)
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    size_t read = 0;
    const size_t threads_count = static_cast<size_t>(ppc::util::GetNumThreads());
    const size_t jstart = (static_cast<size_t>(tid) * b.m) / threads_count;
    const size_t jend = (static_cast<size_t>(tid + 1) * b.m) / threads_count;
    for (size_t j = jstart; j < jend; ++j) {
      const size_t write = c.col_ind[j];
      const size_t cnt = col_nnz[j];
      for (size_t k = 0; k < cnt; ++k) {
        c.value[write + k] = flat_vals[tid][read + k];
        c.row[write + k] = flat_rows[tid][read + k];
      }
      read += cnt;
    }
  }

  return true;
}

bool KulikAMatMulDoubleCcsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace kulik_a_mat_mul_double_ccs
