def gaussian_eliminate(A, b):
    n = len(A)
    m = len(A[0])
    # Tạo ma trận tăng cường M = [A|b]
    M = [[float(x) for x in A[i]] + [float(b[i])] for i in range(n)]

    s = 0        # Số lần hoán đổi hàng
    epsilon = 1e-12

    for k in range(min(n, m)):
        # Partial Pivoting: tìm hàng có |M[i][k]| lớn nhất
        p = max(range(k, n), key=lambda i: abs(M[i][k]))

        if abs(M[p][k]) < epsilon:
            # Không có pivot tại cột k — bỏ qua (hệ có thể vô số nghiệm)
            continue

        # Cảnh báo nếu pivot rất nhỏ (ill-conditioned)
        if abs(M[p][k]) < 1e-7:
            print(f"  [CẢNH BÁO] Pivot tại cột {k} = {M[p][k]:.2e} gần 0 "
                  f"— hệ có thể ill-conditioned (kém ổn định số học).")

        if p != k:
            M[k], M[p] = M[p], M[k]
            s += 1

        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, m + 1):
                M[i][j] -= factor * M[k][j]

    U = [row[:m] for row in M]
    c = [row[m] for row in M]

    x = back_substitution(U, c)
    return U, x, s


def back_substitution(U, c):
    n = len(U)
    m = len(U[0])
    epsilon = 1e-12

    # Kiểm tra vô nghiệm: hàng toàn 0 nhưng c[i] ≠ 0
    for i in range(n):
        if all(abs(U[i][j]) < epsilon for j in range(m)) and abs(c[i]) > epsilon:
            return "Hệ vô nghiệm"

    # Xác định cột pivot và cột tự do
    pivot_cols = []
    pivot_of_row = [-1] * n
    row_idx = 0
    for col_idx in range(m):
        if row_idx < n and abs(U[row_idx][col_idx]) > epsilon:
            pivot_cols.append(col_idx)
            pivot_of_row[row_idx] = col_idx
            row_idx += 1

    free_cols = [j for j in range(m) if j not in pivot_cols]

    # Tính nghiệm riêng x_p (đặt các ẩn tự do = 0)
    x_p = [0.0] * m
    for i in range(n - 1, -1, -1):
        p_col = pivot_of_row[i]
        if p_col != -1:
            sum_val = sum(U[i][j] * x_p[j] for j in range(p_col + 1, m))
            x_p[p_col] = (c[i] - sum_val) / U[i][p_col]

    # Nghiệm duy nhất: trả về list[float] để dùng được trong tính toán
    if not free_cols:
        return x_p

    # Vô số nghiệm: tính các vector cơ sở của không gian nghiệm
    basis_vectors = []
    for f_col in free_cols:
        v = [0.0] * m
        v[f_col] = 1.0   # Gán ẩn tự do tương ứng = 1, các ẩn tự do khác = 0
        for i in range(n - 1, -1, -1):
            p_col = pivot_of_row[i]
            if p_col != -1:
                sum_v = sum(U[i][j] * v[j] for j in range(p_col + 1, m))
                v[p_col] = -sum_v / U[i][p_col]
        basis_vectors.append(v)

    # Trả về (nghiệm riêng, vector cơ sở) — dạng số, dùng được để tính tiếp
    return x_p, basis_vectors
