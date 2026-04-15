"""
solvers.py — Phần 3: Các phương pháp giải hệ phương trình Ax = b

Gộp 3 phương pháp:
  1. Khử Gauss với Partial Pivoting  (Phần 1 — gaussian.py)
  2. Phân rã SVD                      (Phần 2 — decomposition.py)
  3. Lặp Gauss–Seidel                 (Cài đặt mới)

Hàm tiện ích:
  - condition_number_2(A)      : Tính số điều kiện κ₂(A) = σ_max / σ_min
  - is_diagonally_dominant(A)  : Kiểm tra chéo trội chặt hàng
  - relative_error(A, x, b)   : Tính sai số tương đối ‖Ax̂ − b‖₂ / ‖b‖₂
"""

import sys
import os
import math
import copy

# ================================================================
# CẤU HÌNH ĐƯỜNG DẪN IMPORT
# (part3/ và part2/ nằm cùng cấp trong Lab 1/)
# ================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
_part1_dir = os.path.join(_root_dir, "part1")
_part2_dir = os.path.join(_root_dir, "part2")

if _part1_dir not in sys.path:
    sys.path.append(_part1_dir)
if _part2_dir not in sys.path:
    sys.path.append(_part2_dir)

# Import từ Phần 1
from part1.gaussian import gaussian_eliminate

# Import từ Phần 2
from part2.decomposition import svd_decomposition


# ================================================================
# ===================== HÀM HỖ TRỢ ==============================
# ================================================================

def _mat_vec_mul(A, x):
    """Nhân ma trận A (n×n) với vector x (n,). Trả về vector (n,)."""
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def _vec_sub(a, b):
    """Hiệu hai vector: a − b."""
    return [a[i] - b[i] for i in range(len(a))]


def _norm2(v):
    """Chuẩn Euclid (L2) của vector v."""
    return math.sqrt(sum(x * x for x in v))


def _norm_inf(v):
    """Chuẩn vô cùng (max) của vector v."""
    return max(abs(x) for x in v)


def _deep_copy_matrix(A):
    """Tạo bản sao sâu của ma trận (list of lists)."""
    return [row[:] for row in A]


# ================================================================
# ================== 1. GIẢI BẰNG KHỬ GAUSS =====================
# ================================================================

def solve_gauss(A, b):
    """
    Giải Ax = b bằng phương pháp khử Gauss với Partial Pivoting.

    Sử dụng hàm gaussian_eliminate() từ Phần 1 (gaussian.py).

    Tham số:
        A : list of lists — ma trận vuông n×n (khả nghịch)
        b : list           — vector vế phải (n,)

    Trả về:
        x : list[float] — nghiệm của hệ

    Ngoại lệ:
        ValueError nếu hệ không có nghiệm duy nhất.
    """
    # Tạo bản sao để không làm thay đổi dữ liệu gốc
    A_copy = _deep_copy_matrix(A)
    b_copy = b[:]

    _, x, _ = gaussian_eliminate(A_copy, b_copy)

    # gaussian_eliminate trả về tuple (x_p, basis) nếu vô số nghiệm,
    # hoặc string nếu vô nghiệm
    if isinstance(x, str):
        raise ValueError(f"Hệ phương trình vô nghiệm: {x}")
    if isinstance(x, tuple):
        # Vô số nghiệm — lấy nghiệm riêng (particular solution)
        return x[0]

    return x


# ================================================================
# ================== 2. GIẢI BẰNG PHÂN RÃ SVD ===================
# ================================================================

def solve_svd(A, b, sv_threshold=1e-10):
    """
    Giải Ax = b bằng phân rã SVD: x = V Σ⁻¹ Uᵀ b

    Sử dụng hàm svd_decomposition() từ Phần 2 (decomposition.py).

    Công thức:
        A = U Σ Vᵀ
        x = V Σ⁻¹ Uᵀ b  (pseudoinverse)

    Chỉ dùng các singular values σᵢ > sv_threshold để tránh
    chia cho giá trị gần 0, cải thiện ổn định số.

    Tham số:
        A            : list of lists — ma trận vuông n×n
        b            : list           — vector vế phải (n,)
        sv_threshold : float          — ngưỡng bỏ qua singular value nhỏ

    Trả về:
        x : list[float] — nghiệm (least-squares nếu ill-conditioned)
    """
    U, S, Vt = svd_decomposition(A)
    m = len(U)
    n = len(Vt)

    # Bước 1: Tính Uᵀ b  (vector kích thước m)
    Ut_b = [sum(U[i][j] * b[i] for i in range(m)) for j in range(m)]

    # Bước 2: Tính Σ⁻¹ Uᵀ b  (chia cho σᵢ, bỏ qua σ ≈ 0)
    #   Chỉ lấy min(m, n) thành phần tương ứng với singular values
    k = min(m, n, len(S))
    sigma_inv_Ut_b = [0.0] * n
    for i in range(k):
        if S[i] > sv_threshold:
            sigma_inv_Ut_b[i] = Ut_b[i] / S[i]
        # else: giữ = 0 (bỏ qua thành phần này — truncated pseudoinverse)

    # Bước 3: Tính x = Vᵀᵀ (Σ⁻¹ Uᵀ b) = V (Σ⁻¹ Uᵀ b)
    #   Vt là n×n, ta cần V = Vtᵀ, tức x[i] = Σⱼ Vt[j][i] * sigma_inv_Ut_b[j]
    x = [sum(Vt[j][i] * sigma_inv_Ut_b[j] for j in range(n)) for i in range(n)]

    return x


# ================================================================
# ============= 3. GIẢI BẰNG LẶP GAUSS–SEIDEL ==================
# ================================================================

def is_diagonally_dominant(A):
    """
    Kiểm tra ma trận A có chéo trội chặt hàng hay không.

    Điều kiện: |a_ii| > Σ_{j≠i} |a_ij|  với mọi i.

    Đây là điều kiện ĐỦ (sufficient) để Gauss–Seidel hội tụ.

    Tham số:
        A : list of lists — ma trận vuông n×n

    Trả về:
        bool — True nếu chéo trội chặt
    """
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_diag_sum:
            return False
    return True


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Giải Ax = b bằng phương pháp lặp Gauss–Seidel.

    Công thức lặp (theo từng thành phần):
        x_i^(k+1) = (1/a_ii) * ( b_i − Σ_{j<i} a_ij·x_j^(k+1)
                                      − Σ_{j>i} a_ij·x_j^(k) )

    Điểm khác với Jacobi: dùng ngay giá trị x_j mới nhất (j < i)
    trong cùng một vòng lặp → hội tụ nhanh hơn.

    Tham số:
        A        : list of lists — ma trận vuông n×n
        b        : list           — vector vế phải (n,)
        x0       : list or None   — nghiệm khởi tạo (mặc định = vector 0)
        tol      : float          — ngưỡng hội tụ (‖x^(k+1) − x^(k)‖∞ < tol)
        max_iter : int            — số vòng lặp tối đa

    Trả về:
        (x, iterations, converged) :
            x          : list[float] — nghiệm xấp xỉ
            iterations : int         — số vòng lặp đã thực hiện
            converged  : bool        — True nếu đã hội tụ

    Ngoại lệ:
        ValueError nếu phần tử đường chéo a_ii = 0.

    Cảnh báo:
        In cảnh báo nếu ma trận không chéo trội (có thể không hội tụ).
    """
    n = len(A)

    # --- Kiểm tra phần tử đường chéo ---
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            raise ValueError(
                f"Phần tử đường chéo a[{i}][{i}] = 0. "
                f"Gauss–Seidel không thể thực hiện (chia cho 0)."
            )

    # --- Kiểm tra điều kiện hội tụ (chéo trội) ---
    if not is_diagonally_dominant(A):
        print("  [CẢNH BÁO] Ma trận KHÔNG chéo trội chặt hàng. "
              "Gauss–Seidel có thể KHÔNG hội tụ.")

    # --- Khởi tạo nghiệm ---
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]

    # --- Vòng lặp chính ---
    converged = False
    for iteration in range(1, max_iter + 1):
        x_old = x[:]

        for i in range(n):
            # Tính σ = Σ_{j≠i} a_ij * x_j
            # Với j < i: dùng x[j] mới (đã cập nhật trong iteration này)
            # Với j > i: dùng x[j] cũ (chưa cập nhật → chính là x_old[j],
            #            nhưng vì ta update in-place nên x[j] vẫn là giá trị cũ)
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]

            x[i] = (b[i] - sigma) / A[i][i]

        # Kiểm tra hội tụ: ‖x^(k+1) − x^(k)‖∞ < tol
        error = _norm_inf(_vec_sub(x, x_old))
        if error < tol:
            converged = True
            break

    return x, iteration, converged


# ================================================================
# ================= HÀM TIỆN ÍCH (UTILITIES) ====================
# ================================================================

def condition_number_2(A):
    """
    Tính số điều kiện κ₂(A) = σ_max / σ_min (chuẩn spectral).

    Dùng phân rã SVD tự cài đặt (Phần 2).

    Ý nghĩa:
        - κ₂ ≈ 1       : well-conditioned (ổn định)
        - κ₂ >> 1      : ill-conditioned (kém ổn định)
        - κ₂ = ∞       : ma trận suy biến (singular)

    Tham số:
        A : list of lists — ma trận vuông n×n

    Trả về:
        float — số điều kiện κ₂(A), hoặc float('inf') nếu σ_min ≈ 0
    """
    _, S, _ = svd_decomposition(A)

    # Lọc bỏ singular values ≈ 0
    nonzero_sv = [s for s in S if s > 1e-14]

    if not nonzero_sv:
        return float('inf')

    sigma_max = max(nonzero_sv)
    sigma_min = min(nonzero_sv)

    if sigma_min < 1e-14:
        return float('inf')

    return sigma_max / sigma_min


def relative_error(A, x_hat, b):
    """
    Tính sai số tương đối của nghiệm xấp xỉ x̂:
        error = ‖Ax̂ − b‖₂ / ‖b‖₂

    Tham số:
        A     : list of lists — ma trận hệ số n×n
        x_hat : list           — nghiệm xấp xỉ (n,)
        b     : list           — vector vế phải (n,)

    Trả về:
        float — sai số tương đối
    """
    Ax = _mat_vec_mul(A, x_hat)
    residual = _vec_sub(Ax, b)
    norm_b = _norm2(b)

    if norm_b < 1e-15:
        return _norm2(residual)  # Tránh chia cho 0

    return _norm2(residual) / norm_b


# ================================================================
# ========================= DEMO / TEST ==========================
# ================================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("  SOLVERS.PY — Demo giải Ax = b bằng 3 phương pháp")
    print("=" * 60)

    # --- Ví dụ: hệ 3×3 chéo trội ---
    A = [
        [4, 1, 2],
        [3, 5, 1],
        [1, 1, 3]
    ]
    b = [4, 7, 3]

    print(f"\n  Ma trận A:")
    for row in A:
        print(f"    {row}")
    print(f"  Vector b: {b}")
    print(f"  Chéo trội: {is_diagonally_dominant(A)}")

    # --- 1. Khử Gauss ---
    print(f"\n{'─' * 50}")
    print("  [1] Phương pháp khử Gauss (Partial Pivoting)")
    print(f"{'─' * 50}")
    try:
        x_gauss = solve_gauss(A, b)
        err_gauss = relative_error(A, x_gauss, b)
        print(f"  Nghiệm x = {[round(xi, 8) for xi in x_gauss]}")
        print(f"  Sai số tương đối = {err_gauss:.2e}")
    except Exception as e:
        print(f"  Lỗi: {e}")

    # --- 2. Phân rã SVD ---
    print(f"\n{'─' * 50}")
    print("  [2] Phương pháp SVD (pseudoinverse)")
    print(f"{'─' * 50}")
    try:
        x_svd = solve_svd(A, b)
        err_svd = relative_error(A, x_svd, b)
        print(f"  Nghiệm x = {[round(xi, 8) for xi in x_svd]}")
        print(f"  Sai số tương đối = {err_svd:.2e}")
    except Exception as e:
        print(f"  Lỗi: {e}")

    # --- 3. Gauss–Seidel ---
    print(f"\n{'─' * 50}")
    print("  [3] Phương pháp lặp Gauss–Seidel")
    print(f"{'─' * 50}")
    try:
        x_gs, iters, conv = gauss_seidel(A, b)
        err_gs = relative_error(A, x_gs, b)
        print(f"  Nghiệm x = {[round(xi, 8) for xi in x_gs]}")
        print(f"  Số vòng lặp = {iters}")
        print(f"  Hội tụ = {conv}")
        print(f"  Sai số tương đối = {err_gs:.2e}")
    except Exception as e:
        print(f"  Lỗi: {e}")

    # --- Số điều kiện ---
    print(f"\n{'─' * 50}")
    print("  [*] Số điều kiện κ₂(A)")
    print(f"{'─' * 50}")
    try:
        kappa = condition_number_2(A)
        print(f"  κ₂(A) = {kappa:.4f}")
    except Exception as e:
        print(f"  Lỗi: {e}")

    # --- So sánh ---
    print(f"\n{'═' * 60}")
    print("  BẢNG SO SÁNH")
    print(f"{'═' * 60}")
    print(f"  {'Phương pháp':<25} {'Sai số tương đối':<20}")
    print(f"  {'─' * 45}")
    try:
        print(f"  {'Gauss (Pivot)':<25} {err_gauss:<20.2e}")
    except:
        print(f"  {'Gauss (Pivot)':<25} {'N/A':<20}")
    try:
        print(f"  {'SVD':<25} {err_svd:<20.2e}")
    except:
        print(f"  {'SVD':<25} {'N/A':<20}")
    try:
        print(f"  {'Gauss–Seidel':<25} {err_gs:<20.2e}")
    except:
        print(f"  {'Gauss–Seidel':<25} {'N/A':<20}")

    print(f"\n  Done.\n")
