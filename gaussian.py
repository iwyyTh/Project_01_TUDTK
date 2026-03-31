import numpy as np

def gaussian_eliminate(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n, m = A.shape
    epsilon = 1e-12
    M = np.hstack((A, b.reshape(-1, 1)))
    s = 0
    
    for k in range(min(n, m)):
        p = np.argmax(np.abs(M[k:, k])) + k
        if abs(M[p, k]) < epsilon: continue 
        if p != k:
            M[[k, p]] = M[[p, k]]
            s += 1
        for i in range(k + 1, n):
            lik = M[i, k] / M[k, k]
            M[i, k:] = M[i, k:] - lik * M[k, k:]
            
    U = M[:, :m]
    c = M[:, m]
    
    x = back_substitution(U, c)
    return M, x, s

def back_substitution(U, c):
    n, m = U.shape
    epsilon = 1e-12
    
    #Kiểm tra Vô nghiệm
    for i in range(n):
        if np.all(np.abs(U[i, :]) < epsilon) and abs(c[i]) > epsilon:
            return "Hệ vô nghiệm"

    #Tìm các cột chứa Pivot
    pivot_cols = []
    row = 0
    for col in range(m):
        if row < n and abs(U[row, col]) > epsilon:
            pivot_cols.append(col)
            row += 1
            
    #Xác định ẩn tự do
    free_cols = [j for j in range(m) if j not in pivot_cols]
    
    if len(free_cols) == 0 and len(pivot_cols) == m:
        x = np.zeros(m)
        for i in range(n - 1, -1, -1):
            sum_j = np.dot(U[i, i+1:], x[i+1:])
            x[i] = (c[i] - sum_j) / U[i, i]
        return f"Nghiệm duy nhất: {x}"

    x_p = np.zeros(m)
    for i in range(len(pivot_cols) - 1, -1, -1):
        p_col = pivot_cols[i]
        sum_j = np.dot(U[i, p_col + 1:], x_p[p_col + 1:])
        x_p[p_col] = (c[i] - sum_j) / U[i, p_col]

    basis_vectors = []
    for f_col in free_cols:
        v = np.zeros(m)
        v[f_col] = 1 
        for i in range(len(pivot_cols) - 1, -1, -1):
            p_col = pivot_cols[i]
            sum_v = np.dot(U[i, p_col + 1:], v[p_col + 1:])
            v[p_col] = -sum_v / U[i, p_col]
        basis_vectors.append(v)

    res = f"Nghiệm tổng quát: x = {x_p}"
    for idx, v in enumerate(basis_vectors):
        res += f" + t{idx+1} * {v}"
    return res