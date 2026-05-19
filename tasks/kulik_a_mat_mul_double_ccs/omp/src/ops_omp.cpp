#include "kulik_a_mat_mul_double_ccs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "kulik_a_mat_mul_double_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kulik_a_mat_mul_double_ccs {

namespace {
std::vector<size_t> *g_col_thread = nullptr;
std::vector<size_t> *g_col_offset = nullptr;
std::vector<size_t> *g_col_size = nullptr;
}  // namespace

inline void ProcessColumn(size_t j, int tid, const CCS &a, const CCS &b, std::vector<std::vector<double>> &thread_accum,
                          std::vector<std::vector<bool>> &thread_nz, std::vector<std::vector<size_t>> &thread_nnz_rows,
                          std::vector<std::vector<double>> &local_values,
                          std::vector<std::vector<size_t>> &local_rows) {
  auto &accum = thread_accum[tid];
  auto &nz = thread_nz[tid];
  auto &touched_rows = thread_nnz_rows[tid];
  auto &values = local_values[tid];
  auto &rows = local_rows[tid];

  const size_t column_start = values.size();

  for (size_t k = b.col_ind[j]; k < b.col_ind[j + 1]; ++k) {
    const size_t ind = b.row[k];
    const double b_val = b.value[k];
    for (size_t zc = a.col_ind[ind]; zc < a.col_ind[ind + 1]; ++zc) {
      const size_t i = a.row[zc];
      const double a_val = a.value[zc];
      accum[i] += a_val * b_val;
      if (!nz[i]) {
        nz[i] = true;
        touched_rows.push_back(i);
      }
    }
  }

  for (size_t i : touched_rows) {
    if (accum[i] != 0.0) {
      rows.push_back(i);
      values.push_back(accum[i]);
    }
    accum[i] = 0.0;
    nz[i] = false;
  }

  if (g_col_thread != nullptr && g_col_offset != nullptr && g_col_size != nullptr) {
    (*g_col_thread)[j] = static_cast<size_t>(tid);
    (*g_col_offset)[j] = column_start;
    (*g_col_size)[j] = values.size() - column_start;
  }

  touched_rows.clear();
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

  std::vector<std::vector<double>> local_values(num_threads);
  std::vector<std::vector<size_t>> local_rows(num_threads);

  std::vector<std::vector<double>> thread_accum(num_threads, std::vector<double>(a.n, 0.0));
  std::vector<std::vector<bool>> thread_nz(num_threads, std::vector<bool>(a.n, false));
  std::vector<std::vector<size_t>> thread_nnz_rows(num_threads);

  std::vector<size_t> col_thread(b.m, 0);
  std::vector<size_t> col_offset(b.m, 0);
  std::vector<size_t> col_size(b.m, 0);

  g_col_thread = &col_thread;
  g_col_offset = &col_offset;
  g_col_size = &col_size;

#pragma omp parallel for default(none) schedule(static) \
    shared(a, b, thread_accum, thread_nz, thread_nnz_rows, local_values, local_rows)
  for (size_t j = 0; j < b.m; ++j) {
    const int tid = omp_get_thread_num();
    ProcessColumn(j, tid, a, b, thread_accum, thread_nz, thread_nnz_rows, local_values, local_rows);
  }

  g_col_thread = nullptr;
  g_col_offset = nullptr;
  g_col_size = nullptr;

  size_t total_nz = 0;
  for (size_t j = 0; j < b.m; ++j) {
    c.col_ind[j] = total_nz;
    total_nz += col_size[j];
  }
  c.col_ind[b.m] = total_nz;
  c.nz = total_nz;

  c.value.resize(total_nz);
  c.row.resize(total_nz);

#pragma omp parallel for default(none) schedule(static) \
    shared(c, b, local_values, local_rows, col_thread, col_offset, col_size)
  for (size_t j = 0; j < b.m; ++j) {
    const size_t tid = col_thread[j];
    const size_t src = col_offset[j];
    const size_t count = col_size[j];
    const size_t dst = c.col_ind[j];

    for (size_t k = 0; k < count; ++k) {
      c.value[dst + k] = local_values[tid][src + k];
      c.row[dst + k] = local_rows[tid][src + k];
    }
  }

  return true;
}

bool KulikAMatMulDoubleCcsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace kulik_a_mat_mul_double_ccs
