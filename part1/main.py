import numpy as np
from gaussian import gaussian_eliminate
from determinant import determinant

A = np.array([[1, 1, 1], 
                  [2, 2, 2], 
                  [3, 1, 5]])
b = np.array([4, 8, 10])

#Giải hệ
M, x, s = gaussian_eliminate(A, b)
print(f"Nghiệm x: {x}")
print(f"Số lần đổi hàng: {s}")

#Det
det = determinant(A)

#Test
#print(f"Định thức det(A): {det}")
#print(f"Kiểm chứng NumPy Solve: {np.linalg.solve(A, b)}")
#print(f"Kiểm chứng NumPy Det: {np.linalg.det(A)}")