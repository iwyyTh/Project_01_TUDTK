import math
import numpy as np

def _identity(n):
    """Tạo ma trận đơn vị Iₙ (n×n)."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def _mat_mul(A, B):
    """Nhân 2 ma trận A (m×k) × B (k×n) → C (m×n). Thuần Python, không dùng NumPy."""
    m, k_A = len(A), len(A[0])
    k_B, n = len(B), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k_A):
            if A[i][p] == 0: continue
            for j in range(n):
                C[i][j] += A[i][p] * B[p][j]
    return C

def _mat_add(A, B):
    """Cộng 2 ma trận cùng kích thước A + B."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _mat_scale(A, scalar):
    """Nhân ma trận A với một hằng số: scalar × A."""
    n = len(A)
    return [[A[i][j] * scalar for j in range(n)] for i in range(n)]

def _trace(A):
    """Tính vết (trace) của ma trận vuông: tr(A) = Σ aᵢᵢ."""
    return sum(A[i][i] for i in range(len(A)))

def _print_matrix(M, name="M", decimals=4):
    """In ma trận M ra console với tên và số chữ số thập phân tùy chỉnh."""
    print(f"  {name}:")
    for row in M:
        formatted = "  ".join(f"{x:>{decimals+6}.{decimals}f}" for x in row)
        print(f"    [{formatted}]")


# Bước 1: Tìm hệ số đa thức đặc trưng
def faddeev_leverrier(A):
    n = len(A)
    M = [[0.0] * n for _ in range(n)] # M_0 = 0
    coeffs = []
    
    for k in range(1, n + 1):
        # A_k = A * M_{k-1}
        if k == 1:
            Ak = [row[:] for row in A]
        else:
            Ak = _mat_mul(A, M)
            
        # c_{n-k} = -1/k * trace(A_k)
        c = - (1.0 / k) * _trace(Ak)
        coeffs.append(c)
        
        # M_k = A_k + c_{n-k} * I
        cI = _mat_scale(_identity(n), c)
        M = _mat_add(Ak, cI)
        
    # Coeffs đang lưu [c_{n-1}, c_{n-2}, ..., c_0]
    # Lật ngược lại để trả về [c_0, c_1, ..., c_{n-1}]
    return coeffs[::-1]


# Bước 2: Tìm nghiệm đa thức
def solve_quadratic(c1, c0):
    delta = c1**2 - 4*c0
    if delta < 0:
        return [(-c1 + complex(0, math.sqrt(-delta))) / 2.0, (-c1 - complex(0, math.sqrt(-delta))) / 2.0]
    sqrt_delta = math.sqrt(delta)
    return [(-c1 + sqrt_delta) / 2.0, (-c1 - sqrt_delta) / 2.0]

def solve_cubic(c2, c1, c0):
    p = c1 - (c2**2) / 3.0
    q = c0 + (2.0 * (c2**3)) / 27.0 - (c1 * c2) / 3.0
    
    delta = (q**2) / 4.0 + (p**3) / 27.0
    offset = c2 / 3.0
    
    if delta <= 0:
        # Trường hợp Casus Irreducibilis (3 nghiệm thực)
        # Sử dụng phương pháp lượng giác để triệt để rủi ro floating-point (nghiệm ảo)
        r = math.sqrt(- (p**3) / 27.0)
        
        # Trường hợp đặc biệt: p ≈ 0 và q ≈ 0 → 3 nghiệm trùng nhau = -offset
        if r < 1e-14:
            root = -offset
            return [root, root, root]
        
        phi = math.acos(max(-1.0, min(1.0, -q / (2.0 * r))))  # clamp để tránh domain error
        
        magnitude = 2.0 * math.sqrt(-p / 3.0)
        t1 = magnitude * math.cos(phi / 3.0)
        t2 = magnitude * math.cos((phi + 2.0*math.pi) / 3.0)
        t3 = magnitude * math.cos((phi + 4.0*math.pi) / 3.0)
        
        return [t1 - offset, t2 - offset, t3 - offset]
    else:
        # 1 nghiệm thực, 2 nghiệm phức định dạng Cardano
        sqrt_delta = math.sqrt(delta)
        
        def cube_root(val):
            return math.pow(val, 1.0/3.0) if val >= 0 else -math.pow(-val, 1.0/3.0)
            
        u = cube_root(-q/2.0 + sqrt_delta)
        v = cube_root(-q/2.0 - sqrt_delta)
        
        t1 = u + v
        w = complex(-0.5, math.sqrt(3)/2.0)
        w2 = complex(-0.5, -math.sqrt(3)/2.0)
        t2 = u * w + v * w2
        t3 = u * w2 + v * w
        
        return [t1 - offset, t2 - offset, t3 - offset]

# Tính trị riêng bằng QR Iteration cho các ma trận cấp lớn hơn
def compute_eigenvalues_qr(A):
    n = len(A)
    Ak = [row[:] for row in A]
    
    def _transpose(M):
        return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]
        
    for _ in range(1000):
        # Gram-Schmidt
        At = _transpose(Ak)
        Q_cols = []
        R = [[0.0] * n for _ in range(n)]
        for j in range(n):
            v = At[j][:]
            for i in range(len(Q_cols)):
                dot = sum(v[k] * Q_cols[i][k] for k in range(n))
                R[i][j] = dot
                v = [v[k] - dot * Q_cols[i][k] for k in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            R[j][j] = norm
            if norm < 1e-14:
                e = [0.0] * n
                e[j] = 1.0
                Q_cols.append(e)
            else:
                Q_cols.append([x / norm for x in v])
        Q = [[Q_cols[j][i] for j in range(n)] for i in range(n)]
        
        # Ak+1 = R @ Q
        Ak_next = _mat_mul(R, Q)
        
        off_diag = 0.0
        for i in range(1, n):
            for j in range(i):
                off_diag += abs(Ak_next[i][j])
        if off_diag < 1e-10:
            return [Ak_next[i][i] for i in range(n)]
        Ak = Ak_next
        
    return [Ak[i][i] for i in range(n)]

def compute_eigenvalues(A):
    n = len(A)
    
    if n >= 5:
        print(f"  [Info] n={n} >= 5: Sử dụng NumPy (np.linalg.eig)...")
        eigvals, _ = np.linalg.eig(np.array(A, dtype=float))
        return eigvals.tolist()
    
    if n == 4:
        print(f"  [Info] n={n} == 4: Kích hoạt Lặp QR Iteration")
        return compute_eigenvalues_qr(A)
    
    coeffs = faddeev_leverrier(A)
    
    if n == 1:
        roots = [-coeffs[0]]
    elif n == 2:
        print(f"  [Info] Phương trình: λ² + ({coeffs[1]:.4f})λ + ({coeffs[0]:.4f}) = 0")
        roots = solve_quadratic(coeffs[1], coeffs[0])
    elif n == 3:
        print(f"  [Info] Phương trình: λ³ + ({coeffs[2]:.4f})λ² + ({coeffs[1]:.4f})λ + ({coeffs[0]:.4f}) = 0")
        roots = solve_cubic(coeffs[2], coeffs[1], coeffs[0])
        
    return roots


# Bước 3: Tìm vector riêng
def compute_eigenspace(A, eigenvalue, tol=1e-7):
    n = len(A)
    M = [[A[i][j] - (eigenvalue if i == j else 0.0) for j in range(n)] for i in range(n)]

    pivot_cols = []
    row = 0
    for col in range(n):
        max_val = 0.0
        max_row = -1
        for i in range(row, n):
            if abs(M[i][col]) > max_val:
                max_val = abs(M[i][col])
                max_row = i

        if max_val < tol:
            continue

        M[row], M[max_row] = M[max_row], M[row]
        piv = M[row][col]
        M[row] = [x / piv for x in M[row]]
        for i in range(row + 1, n):
            factor = M[i][col]
            M[i] = [M[i][j] - factor * M[row][j] for j in range(n)]

        pivot_cols.append(col)
        row += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols: return []

    eigenspace = []
    # Duyệt qua từng cột tự do để sinh ra các vector riêng độc lập tuyến tính
    for free_col in free_cols:
        x = [0.0] * n
        x[free_col] = 1.0

        for i in range(len(pivot_cols) - 1, -1, -1):
            pc = pivot_cols[i]
            val = -sum(M[i][j] * x[j] for j in range(n) if j != pc)
            x[pc] = val

        norm = math.sqrt(sum(v * v for v in x))
        if norm >= tol:
            eigenspace.append(x)

    return eigenspace

# Các hàm hỗ trợ chéo hóa
def inverse_matrix(A):
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        max_val = abs(aug[col][col])
        max_row = col
        for i in range(col + 1, n):
            if abs(aug[i][col]) > max_val:
                max_val = abs(aug[i][col])
                max_row = i
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Ma trận suy biến.")
        aug[col] = [x / pivot for x in aug[col]]

        for i in range(n):
            if i != col:
                factor = aug[i][col]
                aug[i] = [aug[i][j] - factor * aug[col][j] for j in range(2 * n)]

    return [row[n:] for row in aug]


def diagonalize(A):
    n = len(A)
    eigenvalues = compute_eigenvalues(A)

    real_eigenvalues = []
    for lam in eigenvalues:
        if isinstance(lam, complex):
            if abs(lam.imag) > 1e-6:
                print(f"  [Cảnh báo] Trị riêng phức: {lam} → Có thể không chéo hóa được trên R")
            real_eigenvalues.append(lam.real)
        else:
            real_eigenvalues.append(float(lam))

    P_cols = []
    valid_eigenvalues = []

    unique_eigenvalues = []
    for lam in real_eigenvalues:
        if not any(abs(lam - u) < 1e-5 for u in unique_eigenvalues):
            unique_eigenvalues.append(lam)

    for lam in unique_eigenvalues:
        algebraic_mult = sum(1 for v in real_eigenvalues if abs(v - lam) < 1e-5)
        vectors = compute_eigenspace(A, lam)
        
        used_vectors = vectors[:algebraic_mult]
        
        for vec in used_vectors:
            P_cols.append(vec)
            valid_eigenvalues.append(lam)

    is_diagonalizable = len(P_cols) >= n

    if not P_cols:
        return False, None, None, None, real_eigenvalues, []

    k = min(len(P_cols), n)
    P = [[P_cols[j][i] for j in range(k)] for i in range(n)]
    if k < n:
        for _ in range(n - k):
            for i in range(n): P[i].append(0.0)

    D = [[valid_eigenvalues[i] if (i == j and i < len(valid_eigenvalues)) else 0.0 for j in range(n)] for i in range(n)]

    try:
        P_inv = inverse_matrix(P)
    except ValueError:
        is_diagonalizable = False
        P_inv = None

    return is_diagonalizable, P, D, P_inv, real_eigenvalues, P_cols


def verify_diagonalization(A, P, D, P_inv):
    A_np    = np.array(A,     dtype=float)
    P_np    = np.array(P,     dtype=float)
    D_np    = np.array(D,     dtype=float)
    Pinv_np = np.array(P_inv, dtype=float)

    A_reconstructed = P_np @ D_np @ Pinv_np
    error = np.max(np.abs(A_np - A_reconstructed))
    print(f"  [Verify] max|A - PDP⁻¹| = {error:.2e}  {'✓ OK' if error < 1e-6 else '✗ SAI'}")
    return error < 1e-6


def matrix_power_via_diag(A, k):
    ok, P, D, P_inv, _, _ = diagonalize(A)
    if not ok:
        raise ValueError("Ma trận không chéo hóa được → không thể dùng công thức Aᵏ = PDᵏP⁻¹.")
    n = len(A)
    Dk = [[D[i][i] ** k if i == j else 0.0 for j in range(n)] for i in range(n)]
    Ak = _mat_mul(_mat_mul(P, Dk), P_inv)
    return Ak


# Nhập ma trận & xuất kết quả
def input_matrix():
    n = int(input("Nhập kích thước ma trận vuông n: "))
    print(f"Nhập {n} dòng, mỗi dòng {n} số cách nhau bởi dấu cách:")
    A = []
    for i in range(n):
        row = list(map(float, input(f"  Dòng {i+1}: ").split()))
        if len(row) != n:
            raise ValueError(f"Dòng {i+1} có {len(row)} phần tử, cần {n}.")
        A.append(row)
    return A


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    while True:
        print("\n" + "=" * 60)
        print("CHÉO HÓA MA TRẬN (A = PDP⁻¹)")
        print("=" * 60)
        print("  [1] Nhập ma trận")
        print("  [2] Tính Aᵏ (lũy thừa ma trận)")
        print("  [0] Thoát")
        choice = input("\nChọn: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            try:
                A = input_matrix()
                print()
                _print_matrix(A, "A (ma trận nhập)")

                is_diag, P, D, P_inv, eigvals, eigvecs = diagonalize(A)

                print(f"\n  Tất cả trị riêng: {[round(v, 4) for v in eigvals]}")

                print("  Các cặp (trị riêng, vector riêng) tìm được:")
                for i, vec in enumerate(eigvecs):
                    lam = D[i][i] if (is_diag and D) else eigvals[i] if i < len(eigvals) else "?"
                    vec_str = "[" + ", ".join(f"{v:8.4f}" for v in vec) + "]"
                    if isinstance(lam, float):
                        print(f"    λ = {lam:>8.4f}  →  v = {vec_str}")
                    else:
                        print(f"    λ = {lam}  →  v = {vec_str}")

                print(f"\n  Chéo hóa được: {'Có' if is_diag else 'Không'}")

                if is_diag and P and P_inv:
                    _print_matrix(P,     "P  (mỗi CỘT là một vector riêng)")
                    _print_matrix(D,     "D  (trị riêng trên đường chéo)")
                    _print_matrix(P_inv, "P⁻¹")
                    verify_diagonalization(A, P, D, P_inv)

            except Exception as e:
                print(f"  Lỗi: {e}")

        elif choice == "2":
            try:
                A = input_matrix()
                k = int(input("Nhập số mũ k: "))
                print()

                Ak = matrix_power_via_diag(A, k)
                _print_matrix(A, "A")
                _print_matrix(Ak, f"A^{k} = PDᵏP⁻¹")

                # So sánh với NumPy
                Ak_np = np.linalg.matrix_power(np.array(A, dtype=float), k)
                err = np.max(np.abs(np.array(Ak) - Ak_np))
                print(f"\n  So sánh với NumPy: max error = {err:.2e}  {'OK' if err < 1e-6 else 'SAI'}")

            except Exception as e:
                print(f"  Lỗi: {e}")

