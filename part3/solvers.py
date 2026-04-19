import sys
import os
import math
import copy

# Cấu hình đường dẫn import để python nhận diện được module
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)

if _root_dir not in sys.path:
    sys.path.append(_root_dir)

# Import từ Phần 1
from part1.gaussian import gaussian_eliminate

# Import từ Phần 2
from part2.decomposition import svd_decomposition

# Các hàm hỗ trợ

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


# 1. Giải hệ phương trình Ax = b bằng khử Gauss
def solve_gauss(A, b):
    A_copy = _deep_copy_matrix(A)
    b_copy = b[:]

    _, x, _ = gaussian_eliminate(A_copy, b_copy)

    if isinstance(x, str):
        if "Nghiệm duy nhất" in x or "Nghiệm tổng quát" in x:
            import ast
            start = x.find("[")
            end = x.find("]", start) + 1
            if start != -1 and end != 0:
                try:
                    return ast.literal_eval(x[start:end])
                except:
                    pass
        raise ValueError(f"Hệ phương trình báo lỗi: {x}")
        
    if isinstance(x, tuple):
        return x[0]

    return x


# 2. Giải hệ phương trình bằng phân rã SVD (SVD decomposition)
def solve_svd(A, b, sv_threshold=1e-10):
    U, S, Vt = svd_decomposition(A)
    m = len(U)
    n = len(Vt)

    Ut_b = [sum(U[i][j] * b[i] for i in range(m)) for j in range(m)]

    k = min(m, n, len(S))
    sigma_inv_Ut_b = [0.0] * n
    for i in range(k):
        if S[i] > sv_threshold:
            sigma_inv_Ut_b[i] = Ut_b[i] / S[i]

    x = [sum(Vt[j][i] * sigma_inv_Ut_b[j] for j in range(n)) for i in range(n)]

    return x


# 3. Phép lặp Gauss-Seidel

def is_diagonally_dominant(A):
    """ Kiểm tra xem ma trận có chéo trội chặt không """
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_diag_sum:
            return False
    return True


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """ Giải hệ phương trình bằng lặp Gauss-Seidel """
    n = len(A)

    # Kiểm tra đường chéo
    for i in range(n):
        if abs(A[i][i]) < 1e-15:
            raise ValueError(f"Phần tử đường chéo a[{i}][{i}] = 0 (Gauss-Seidel lỗi do chia cho 0).")

    # Cảnh báo nếu không chéo trội
    if not is_diagonally_dominant(A):
        print("  [Cảnh báo] Ma trận không chéo trội chặt hàng. Có thể không hội tụ.")

    # --- Khởi tạo nghiệm ---
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]

    # Bắt đầu vòng lặp
    converged = False
    for iteration in range(1, max_iter + 1):
        x_old = x[:]

        for i in range(n):
            # Tính tổng các thành phần khác i
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]

            x[i] = (b[i] - sigma) / A[i][i]

        error = _norm_inf(_vec_sub(x, x_old))
        if error < tol:
            converged = True
            break

    return x, iteration, converged


# Các hàm đánh giá độ chính xác (Utility)

def condition_number_2(A):
    """ Tính số điều kiện κ₂(A) = σ_max / σ_min """
    _, S, _ = svd_decomposition(A)

    nonzero_sv = [s for s in S if s > 1e-14]

    if not nonzero_sv:
        return float('inf')

    sigma_max = max(nonzero_sv)
    sigma_min = min(nonzero_sv)

    if sigma_min < 1e-14:
        return float('inf')

    return sigma_max / sigma_min


def relative_error(A, x_hat, b):
    """ Tính sai số tương đối: ‖Ax - b‖ / ‖b‖ """
    Ax = _mat_vec_mul(A, x_hat)
    residual = _vec_sub(Ax, b)
    norm_b = _norm2(b)

    if norm_b < 1e-15:
        return _norm2(residual) 

    return _norm2(residual) / norm_b


# Chạy demo

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
