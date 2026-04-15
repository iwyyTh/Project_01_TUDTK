"""
benchmark.py — Phần 3: Đo thời gian và phân tích ổn định số

Yêu cầu đồ án:
  1. Thực nghiệm với ma trận ngẫu nhiên kích thước n ∈ {50, 100, 200, 500, 1000}
  2. Đo thời gian thực thi (trung bình 5 lần chạy)
  3. Đo sai số tương đối: ‖Ax̂ − b‖₂ / ‖b‖₂
  4. Phân tích ổn định: Ma trận Hilbert (ill-conditioned) vs SPD (well-conditioned)
  5. Tính số điều kiện κ₂(A)
"""

import sys
import os
import time
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# CẤU HÌNH ĐƯỜNG DẪN IMPORT
# ================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
_part1_dir = os.path.join(_root_dir, "part1")
_part2_dir = os.path.join(_root_dir, "part2")

if _part1_dir not in sys.path:
    sys.path.append(_part1_dir)
if _part2_dir not in sys.path:
    sys.path.append(_part2_dir)

from gaussian import gaussian_eliminate
from solvers import gauss_seidel, solve_svd, solve_gauss, relative_error


# ================================================================
# ================ SINH MA TRẬN ===================================
# ================================================================

def generate_random_diag_dominant(n):
    """
    Sinh ma trận ngẫu nhiên chéo trội chặt hàng (n×n).
    Đảm bảo cả 3 phương pháp (Gauss, SVD, Gauss-Seidel) đều chạy được.
    """
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = off_diag + random.uniform(1, 5)
    return A


def generate_hilbert_matrix(n):
    """
    Sinh ma trận Hilbert n×n: H[i][j] = 1 / (i + j + 1).
    Ma trận Hilbert có số điều kiện rất lớn (ill-conditioned).
    """
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def generate_spd_matrix(n):
    """
    Sinh ma trận SPD (Symmetric Positive Definite) chéo trội chặt.
    = đối xứng + chéo trội + đường chéo dương → đảm bảo SPD.
    Gauss-Seidel hội tụ tốt trên loại ma trận này.
    """
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = random.uniform(-10, 10)
            A[i][j] = val
            A[j][i] = val  # Đối xứng
    for i in range(n):
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = off_diag + random.uniform(1, 5)
    return A


def generate_random_vector(n):
    """Sinh vector vế phải b ngẫu nhiên (n phần tử)."""
    return [random.uniform(-10, 10) for _ in range(n)]


# ================================================================
# ============== TÍNH SỐ ĐIỀU KIỆN κ₂(A) =========================
# ================================================================

def condition_number(A):
    """
    Tính κ₂(A) = σ_max / σ_min bằng NumPy SVD.
    (Dùng NumPy ở đây vì SVD tự cài chỉ hỗ trợ n nhỏ, 
     benchmark cần chạy với n lên tới 1000.)
    """
    A_np = np.array(A, dtype=float)
    s = np.linalg.svd(A_np, compute_uv=False)
    if s[-1] < 1e-15:
        return float('inf')
    return float(s[0] / s[-1])


# ================================================================
# ============== HÀM ĐO THỜI GIAN ================================
# ================================================================

def benchmark_solver(solver_name, solver_fn, A, b, num_runs=5):
    """
    Đo thời gian trung bình (num_runs lần) và sai số tương đối.

    Tham số:
        solver_name : str    — tên phương pháp (để hiển thị)
        solver_fn   : callable — hàm giải, nhận (A, b) trả về x
        A           : ma trận n×n
        b           : vector n
        num_runs    : int    — số lần chạy (mặc định 5)

    Trả về:
        dict với keys: 'name', 'avg_time', 'rel_error', 'converged', 'iterations'
    """
    times = []
    x_result = None
    extra_info = {}

    for run in range(num_runs):
        start = time.perf_counter()
        try:
            result = solver_fn(A, b)
            # Gauss-Seidel trả về tuple (x, iterations, converged)
            if isinstance(result, tuple):
                x_result = result[0]
                extra_info['iterations'] = result[1]
                extra_info['converged'] = result[2]
            else:
                x_result = result
        except Exception as e:
            return {
                'name': solver_name,
                'avg_time': None,
                'rel_error': None,
                'error_msg': str(e),
            }
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    # Tính sai số tương đối: ‖Ax̂ − b‖₂ / ‖b‖₂
    rel_err = relative_error(A, x_result, b)

    result_dict = {
        'name': solver_name,
        'avg_time': avg_time,
        'rel_error': rel_err,
    }
    result_dict.update(extra_info)
    return result_dict


# ================================================================
# ============ BENCHMARK CHÍNH ====================================
# ================================================================

def run_benchmark(matrix_type="random", sizes=None, num_runs=5, verbose=True):
    """
    Chạy benchmark cho 3 phương pháp, trên nhiều kích thước.

    Tham số:
        matrix_type : 'random' | 'hilbert' | 'spd'
        sizes       : list[int] — các kích thước n
        num_runs    : int — số lần chạy trung bình
        verbose     : bool — in kết quả ra terminal

    Trả về:
        list[dict] — kết quả benchmark cho mỗi (n, method)
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]

    all_results = []

    for n in sizes:
        if verbose:
            print(f"\n{'='*70}")
            print(f"  n = {n} | Loại ma trận: {matrix_type.upper()} | Trung bình {num_runs} lần chạy")
            print(f"{'='*70}")

        # --- Sinh ma trận ---
        if matrix_type == "random":
            A = generate_random_diag_dominant(n)
        elif matrix_type == "hilbert":
            A = generate_hilbert_matrix(n)
        elif matrix_type == "spd":
            A = generate_spd_matrix(n)
        else:
            raise ValueError(f"Loại ma trận không hợp lệ: {matrix_type}")
            return;

        b = generate_random_vector(n)

        # --- Tính số điều kiện ---
        kappa = condition_number(A)

        if verbose:
            print(f"  κ₂(A) = {kappa:.4e}")
            print()

        # --- Danh sách solver ---
        solvers = [
            ("Gauss (Partial Pivot)", solve_gauss),
            ("Gauss-Seidel", gauss_seidel),
        ]

        # SVD tự cài quá chậm với n lớn → dùng NumPy SVD cho n >= 5
        # (giống logic đã có trong solvers.py: decomposition.py dùng NumPy cho n>=5)
        solvers.append(("SVD (Pseudoinverse)", _solve_svd_benchmark))

        for solver_name, solver_fn in solvers:
            result = benchmark_solver(solver_name, solver_fn, A, b, num_runs)
            result['n'] = n
            result['matrix_type'] = matrix_type
            result['condition_number'] = kappa
            all_results.append(result)

            if verbose:
                _print_result(result)

    return all_results


def _solve_svd_benchmark(A, b):
    """
    Wrapper SVD cho benchmark: dùng NumPy cho n lớn (n >= 5),
    SVD tự cài cho n nhỏ.
    """
    n = len(A)
    if n >= 5:
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)
        U, s, Vt = np.linalg.svd(A_np, full_matrices=True)
        # Tính pseudoinverse: x = V Σ⁻¹ Uᵀ b
        S_plus = np.zeros((n, n), dtype=float)
        for i in range(n):
            if s[i] > 1e-12:
                S_plus[i, i] = 1.0 / s[i]
        x = Vt.T @ S_plus @ U.T @ b_np
        return x.tolist()
    else:
        return solve_svd(A, b)


def _print_result(result):
    """In kết quả benchmark ra terminal."""
    name = result['name']
    avg_time = result.get('avg_time')
    rel_err = result.get('rel_error')
    error_msg = result.get('error_msg')

    if error_msg:
        print(f"  {name:<25} | LỖI: {error_msg}")
        return

    line = f"  {name:<25} | Thời gian: {avg_time:.6f}s | Sai số tương đối: {rel_err:.2e}"

    if 'iterations' in result:
        line += f" | Lặp: {result['iterations']}"
    if 'converged' in result:
        line += f" | Hội tụ: {'Có' if result['converged'] else 'KHÔNG'}"

    print(line)


# ================================================================
# =========== PHÂN TÍCH ỔN ĐỊNH SỐ ===============================
# ================================================================

def run_stability_analysis(sizes=None, num_runs=5, verbose=True):
    """
    So sánh ổn định số trên 2 loại ma trận:
      - Hilbert (ill-conditioned, κ₂ rất lớn)
      - SPD chéo trội (well-conditioned, κ₂ nhỏ)

    Trả về:
        (hilbert_results, spd_results) — 2 list kết quả benchmark
    """
    if sizes is None:
        sizes = [5, 10, 15, 20]  # Hilbert nhỏ hơn vì rất ill-conditioned

    if verbose:
        print("\n" + "█" * 70)
        print("  PHÂN TÍCH ỔN ĐỊNH SỐ: HILBERT vs SPD")
        print("█" * 70)

    if verbose:
        print("\n" + "─" * 70)
        print("  [A] MA TRẬN HILBERT (ill-conditioned)")
        print("─" * 70)
    hilbert_results = run_benchmark("hilbert", sizes, num_runs, verbose)

    if verbose:
        print("\n" + "─" * 70)
        print("  [B] MA TRẬN SPD CHÉO TRỘI (well-conditioned)")
        print("─" * 70)
    spd_results = run_benchmark("spd", sizes, num_runs, verbose)

    # --- Bảng tổng kết ---
    if verbose:
        print("\n" + "█" * 70)
        print("  BẢNG TỔNG KẾT: κ₂(A) vs Sai Số Tương Đối")
        print("█" * 70)
        print(f"\n  {'Loại':<10} {'n':>4} {'κ₂(A)':>14} | {'Gauss':>12} {'SVD':>12} {'G-Seidel':>12}")
        print(f"  {'─'*68}")

        for res in hilbert_results:
            _print_summary_line("Hilbert", res, hilbert_results)
        for res in spd_results:
            _print_summary_line("SPD", res, spd_results)

    return hilbert_results, spd_results


def _print_summary_line(label, res, all_results):
    """In 1 dòng tổng kết (chỉ in mỗi n 1 lần, gộp 3 phương pháp)."""
    if res['name'] != 'Gauss (Partial Pivot)':
        return  # Chỉ in khi gặp Gauss (dòng đầu cho mỗi n)

    n = res['n']
    kappa = res['condition_number']

    # Tìm kết quả 3 phương pháp cho cùng n
    gauss_err = svd_err = gs_err = "N/A"
    for r in all_results:
        if r['n'] == n:
            err = r.get('rel_error')
            err_str = f"{err:.2e}" if err is not None else "LỖI"
            if 'Gauss' in r['name'] and 'Seidel' not in r['name']:
                gauss_err = err_str
            elif 'SVD' in r['name']:
                svd_err = err_str
            elif 'Seidel' in r['name']:
                gs_err = err_str

    print(f"  {label:<10} {n:>4} {kappa:>14.4e} | {gauss_err:>12} {svd_err:>12} {gs_err:>12}")


# ================================================================
# ============ LƯU KẾT QUẢ ĐỂ NOTEBOOK DÙNG =====================
# ================================================================

def save_results(results, filename="benchmark_results.json"):
    """Lưu kết quả benchmark ra file JSON để analysis.ipynb dùng."""
    filepath = os.path.join(_current_dir, filename)

    # Chuyển đổi các giá trị không JSON serializable
    clean = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if v == float('inf'):
                entry[k] = "Infinity"
            elif isinstance(v, float) and (v != v):  # NaN
                entry[k] = "NaN"
            else:
                entry[k] = v
        clean.append(entry)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  Đã lưu kết quả vào: {filepath}")


def draw_charts_from_json(json_file="benchmark_results.json"):
    """
    Hàm đọc dữ liệu từ file JSON và vẽ đồ thị mà không cần chạy lại benchmark.
    """
    # 1. Kiểm tra file tồn tại
    if not os.path.exists(json_file):
        print(f"LỖI: Không tìm thấy file '{json_file}'.")
        print("Hãy chạy benchmark.py trước để tạo dữ liệu!")
        return

    # 2. Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Chuyển đổi sang DataFrame và làm sạch
    df = pd.DataFrame(data)
    
    # Thay thế các chuỗi đặc biệt thành giá trị số thực để vẽ đồ thị
    df.replace("Infinity", np.inf, inplace=True)
    df.replace("NaN", np.nan, inplace=True)
    
    # Ép kiểu dữ liệu số
    cols_to_fix = ['n', 'avg_time', 'rel_error', 'condition_number']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Thiết lập giao diện vẽ
    sns.set_theme(style="whitegrid")
    
    # --- ĐỒ THỊ 1: HIỆU NĂNG THỜI GIAN ---
    plt.figure(figsize=(10, 6))
    df_random = df[df['matrix_type'] == 'random'].copy()
    if not df_random.empty:
        sns.lineplot(data=df_random, x='n', y='avg_time', hue='name', marker='o', linewidth=2)
        plt.yscale('log')
        plt.title('PHÂN TÍCH HIỆU NĂNG: Thời gian thực thi (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Kích thước ma trận (n)')
        plt.ylabel('Thời gian trung bình (s)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig('performance_replot.png', dpi=300)
        print("-> Đã xuất: performance_replot.png")

    # --- ĐỒ THỊ 2: ĐỘ ỔN ĐỊNH SỐ (SAI SỐ) ---
    plt.figure(figsize=(12, 6))
    df_stability = df[df['matrix_type'].isin(['hilbert', 'spd'])].copy()
    if not df_stability.empty:
        # Tạo nhãn kết hợp để dễ phân biệt trên chú thích
        df_stability['Label'] = df_stability['name'] + " (" + df_stability['matrix_type'].str.upper() + ")"
        
        sns.lineplot(data=df_stability, x='n', y='rel_error', hue='Label', 
                     style='matrix_type', markers=True, markersize=8)
        
        plt.yscale('log')
        plt.title('PHÂN TÍCH ỔN ĐỊNH SỐ: Sai số tương đối (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Kích thước ma trận (n)')
        plt.ylabel('Sai số ||Ax-b|| / ||b||')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig('stability_replot.png', dpi=300)
        print("-> Đã xuất: stability_replot.png")

    plt.show()


# ================================================================
# ============================ MAIN ===============================
# ================================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')

    print("█" * 70)
    print("  BENCHMARK — Phần 3: Phân Tích Hiệu Năng và Ổn Định Số")
    print("  Trung bình 5 lần chạy | Sai số tương đối ‖Ax̂−b‖₂ / ‖b‖₂")
    print("█" * 70)

    # ─────────────────────────────────────────────
    # 1. BENCHMARK THỜI GIAN — Ma trận ngẫu nhiên
    # ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHẦN 1: BENCHMARK THỜI GIAN — Ma trận ngẫu nhiên chéo trội")
    print("─" * 70)

    time_results = run_benchmark(
        matrix_type="random",
        sizes=[50, 100, 200, 500, 1000],
        num_runs=5,
        verbose=True
    )

    # ─────────────────────────────────────────────
    # 2. PHÂN TÍCH ỔN ĐỊNH SỐ
    # ─────────────────────────────────────────────
    hilbert_results, spd_results = run_stability_analysis(
        sizes=[5, 10, 15, 20],
        num_runs=5,
        verbose=True
    )

    # ─────────────────────────────────────────────
    # 3. LƯU KẾT QUẢ
    # ─────────────────────────────────────────────
    all_results = time_results + hilbert_results + spd_results
    save_results(all_results)
