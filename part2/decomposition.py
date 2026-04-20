import sys
import os
import math
import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
if _root_dir not in sys.path:
    sys.path.append(_root_dir)

from part2.diagonalization import diagonalize, compute_eigenvalues

#Nhân 2 ma trận
def _mat_mul(A, B):
    m, k, n = len(A), len(B), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            if A[i][p] == 0:
                continue
            for j in range(n):
                C[i][j] += A[i][p] * B[p][j]
    return C

#A chuyển vị
def _transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _print_matrix(M, name="M", decimals=4):
    print(f"\n  {name}:")
    for row in M:
        formatted = "  ".join(f"{x:>{decimals+6}.{decimals}f}" for x in row)
        print(f"    [{formatted}]")

#Giải thuật Gram - schimdt
def _gram_schmidt_extend(existing_vecs, m):
    """
    Bổ sung các vector trực chuẩn bằng Gram–Schmidt để có đủ m vector.
    Dùng khi U cần thêm cột (m > r, với r = rank).
    """
    basis = [v[:] for v in existing_vecs]
    standard = [[1.0 if i == k else 0.0 for i in range(m)] for k in range(m)]
    for e in standard:
        if len(basis) >= m:
            break
        v = e[:]
        for b in basis:
            dot = sum(v[i] * b[i] for i in range(m))
            v = [v[i] - dot * b[i] for i in range(m)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 1e-10:
            basis.append([x / norm for x in v])
    return basis



#Phân rã SVD
def svd_decomposition(A):
    """
    Tham số:
        A: list of lists — ma trận m x n
    Trả về:
        U:  list of lists — ma trận trực giao m x m
        S:  list — singular values (giảm dần)
        Vt: list of lists — ma trận Vt (n x n)
    """
    # Kiểm tra đầu vào cơ bản
    if len(A) == 0 or len(A[0]) == 0:
        raise ValueError("Ma trận rỗng.")
    m = len(A)
    n = len(A[0])
    if any(len(row) != n for row in A):
        raise ValueError("Các dòng không cùng độ dài.")
    At = _transpose(A)               # n x m
    AtA = _mat_mul(At, A)            # n x n  đối xứng bán xác định dương
    # -------------------------------------------------------
    # Bước 1: Chéo hóa 
    # -------------------------------------------------------
    _, P, D, _, eigenvalues, eigenvecs = diagonalize(AtA)

    # Sắp xếp giảm dần theo trị riêng
    pairs = sorted(zip(eigenvalues, eigenvecs), key=lambda x: x[0], reverse=True)
    eigenvalues_sorted = [p[0] for p in pairs]
    eigenvecs_sorted   = [p[1] for p in pairs]

    # -------------------------------------------------------
    # Bước 2: Tính singular values
    # -------------------------------------------------------
    singular_values = [math.sqrt(max(lam, 0.0)) for lam in eigenvalues_sorted]
    r = sum(1 for sv in singular_values if sv > 1e-7)   # rank thực sự (tăng dung sai để tránh noise 1e-8)

    
    # -------------------------------------------------------
    # Bước 3: Xây dựng V (n x n), mỗi cột là một vector riêng ĐÃ CHUẨN HÓA
    # -------------------------------------------------------
    
    def _normalize(v):
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm > 1e-14 else v

    eigenvecs_normalized = [_normalize(v) for v in eigenvecs_sorted]
    V = [[eigenvecs_normalized[j][i] for j in range(n)] for i in range(n)]

    # -------------------------------------------------------
    # Bước 4: Tính các cột của U
    # -------------------------------------------------------
    
    U_cols = []
    for j in range(r):
        vj  = [V[i][j] for i in range(n)]                              # cột j của V
        Avj = [sum(A[i][k] * vj[k] for k in range(n)) for i in range(m)]
        uj  = [x / singular_values[j] for x in Avj]
        U_cols.append(uj)

    # -------------------------------------------------------
    # Bước 5: Bổ sung cột còn thiếu của U bằng Gram–Schmidt
    # -------------------------------------------------------
    
    U_cols = _gram_schmidt_extend(U_cols, m)
    U = [[U_cols[j][i] for j in range(m)] for i in range(m)]          # m x m

    Vt = _transpose(V)                                                  # n x n

    return U, singular_values, Vt


def verify_svd(A, U, S, Vt):
    #Kiểm chứng bằng NumPy.
    A_np, U_np, Vt_np = (np.array(M, dtype=float) for M in [A, U, Vt])
    m, n = A_np.shape

    Sigma = np.zeros((m, n))
    for i, sv in enumerate(S):
        if i < min(m, n):
            Sigma[i, i] = sv

    err_A = np.max(np.abs(A_np - U_np @ Sigma @ Vt_np))
    err_U = np.max(np.abs(U_np.T @ U_np - np.eye(m)))
    err_V = np.max(np.abs(Vt_np @ Vt_np.T - np.eye(n)))

    print(f"  [SVD Verify] max|A - UΣVt|   = {err_A:.2e}  {'OK' if err_A < 1e-6 else 'SAI'}")
    print(f"  [SVD Verify] U trực giao      = {err_U:.2e}  {'OK' if err_U < 1e-6 else 'SAI'}")
    print(f"  [SVD Verify] V trực giao      = {err_V:.2e}  {'OK' if err_V < 1e-6 else 'SAI'}")
    return err_A < 1e-6


# ================================================================
# ==================== NHẬP MA TRẬN & XUẤT KẾT QUẢ ==============
# ================================================================
def input_matrix_svd():
    """Nhập ma trận m×n từ bàn phím (SVD chấp nhận mọi kích thước)."""
    m = int(input("Nhập số dòng m: "))
    n = int(input("Nhập số cột n: "))
    print(f"Nhập {m} dòng, mỗi dòng {n} số cách nhau bởi dấu cách:")
    A = []
    for i in range(m):
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
        print("PHÂN RÃ SVD (A = UΣVᵀ)")
        print("=" * 60)
        print("  [1] Nhập ma trận")
        print("  [0] Thoát")
        choice = input("\nChọn: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            try:
                A = input_matrix_svd()
                m, n = len(A), len(A[0])
                print()

                U, S, Vt = svd_decomposition(A)

                _print_matrix(A, "A (ma trận nhập)")
                print(f"\n  Singular values: {[round(s, 6) for s in S]}")
                print(f"  Rank = {sum(1 for s in S if s > 1e-7)}")

                _print_matrix(U, "U (left singular vectors, m x m)")

                Sigma_display = [[0.0]*n for _ in range(m)]
                for i in range(min(m, n)):
                    if i < len(S):
                        Sigma_display[i][i] = S[i]
                _print_matrix(Sigma_display, "Σ (singular values trên đường chéo)")
                _print_matrix(Vt, "Vt (right singular vectors, n×n)")

                verify_svd(A, U, S, Vt)

            except Exception as e:
                print(f"  Lỗi: {e}")

