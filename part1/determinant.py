from gaussian import gaussian_eliminate

def determinant(A):
    n = len(A)
    for row in A:
        if len(row) != n:
            return "Ma trận không vuông"
    
    b = [0.0] * n
    
    M, _, s = gaussian_eliminate(A, b)
    
    det_val = (-1.0) ** s
    for i in range(n):
        det_val *= M[i][i]
        
    return det_val