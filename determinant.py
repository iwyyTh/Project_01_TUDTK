import numpy as np
from gaussian import gaussian_eliminate

def determinant(A):
    """
    1.1.3. Tính Định Thức qua Khử Gauss
    Công thức: det(A) = (-1)^s * tích các phần tử đường chéo u_ii
    """
    n, m = A.shape
    if n != m:
        return "Ma trận không vuông"
    
    # Tạo vector b toàn 0 để chạy hàm gaussian_eliminate
    b_dummy = np.zeros(n)
    
    # Khử Gauss để lấy ma trận sau khử (M) và số lần đổi hàng (s)
    M, _, s = gaussian_eliminate(A, b_dummy)
    
    # Định thức là tích các phần tử trên đường chéo chính của phần A trong M
    det_val = ((-1)**s)
    for i in range(n):
        det_val *= M[i, i]
        
    return det_val