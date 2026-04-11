def gaussian_eliminate(A, b):
    # Tạo ma trận bổ sung M = [A|b]
    n = len(A)
    m = len(A[0])
    M = []
    for i in range(n):
        row = [float(x) for x in A[i]] + [float(b[i])]
        M.append(row)

    s = 0  # Biến đếm số lần đổi hàng
    epsilon = 1e-12

    for k in range(min(n, m)):
        p = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[p][k]):
                p = i

        if abs(M[p][k]) < epsilon:
            continue

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

    return M, x, s


def back_substitution(U, c):
    n = len(U)
    m = len(U[0])
    epsilon = 1e-12

    # Kiểm tra vô nghiệm
    for i in range(n):
        row_is_zero = True
        for j in range(m):
            if abs(U[i][j]) > epsilon:
                row_is_zero = False
                break
        if row_is_zero and abs(c[i]) > epsilon:
            return "Hệ vô nghiệm"

    # Tìm các cột chứa pivot
    pivot_cols = []
    pivot_of_row = [-1] * n
    row_idx = 0
    for col_idx in range(m):
        if row_idx < n and abs(U[row_idx][col_idx]) > epsilon:
            pivot_cols.append(col_idx)
            pivot_of_row[row_idx] = col_idx
            row_idx += 1

    free_cols = [j for j in range(m) if j not in pivot_cols]

    # Tìm Nghiệm riêng (x_p)
    x_p = [0.0] * m
    for i in range(n - 1, -1, -1):
        p_col = pivot_of_row[i]
        if p_col != -1:
            sum_val = 0
            for j in range(p_col + 1, m):
                sum_val += U[i][j] * x_p[j]
            x_p[p_col] = (c[i] - sum_val) / U[i][p_col]

    # Không có ẩn tự do -> Nghiệm duy nhất
    if not free_cols:
        x_p_rounded = [round(val, 10) for val in x_p]
        return f"Nghiệm duy nhất: x = {x_p_rounded}"

    # Tìm vector cơ sở cho hệ nghiệm
    basis_vectors = []
    for f_col in free_cols:
        v = [0.0] * m
        v[f_col] = 1.0  # Gán ẩn tự do = 1
        for i in range(n - 1, -1, -1):
            p_col = pivot_of_row[i]
            if p_col != -1:
                sum_v = 0
                for j in range(p_col + 1, m):
                    sum_v += U[i][j] * v[j]
                v[p_col] = -sum_v / U[i][p_col]
        basis_vectors.append(v)

    # Công thức nghiệm tổng quát
    x_p_rounded = [round(val, 10) for val in x_p]
    res = f"Nghiệm tổng quát: x = {x_p_rounded}"
    for idx, v in enumerate(basis_vectors):
        v_rounded = [round(val, 10) for val in v]
        res += f" + t{idx+1} * {v_rounded}"
    return res
