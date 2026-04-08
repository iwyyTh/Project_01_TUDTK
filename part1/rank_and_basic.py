from inverse import get_partial_pivot
import copy
from fractions import Fraction


def get_rref(matrix: list) -> tuple[list, list]:
    """
    Tạo ma trận rref đề hỗ trợ hàm rank_and_basic

    Args:
        matrix : ma trận cần chuyển về dạng rref

    Returns:
        tuple gồm :
        - stranfer_matrix : ma trận chuyển đổi
        - pivot_cols : danh sách các cột pivot
    """

    stranfer_matrix = [[Fraction(x) for x in row] for row in matrix]
    rows = len(stranfer_matrix)
    cols = len(stranfer_matrix[0])
    pivot_cols = []

    for row in stranfer_matrix:
        if len(row) != cols:
            raise ValueError("Ma trận không đúng định dạng")

    r = 0
    for c in range(cols):
        if r >= rows:
            break

        # Tìm dòng có abs lớn nhất tại cột c
        pivot_row, value = get_partial_pivot(stranfer_matrix, c, r)
        if value == 0:
            continue

        stranfer_matrix[pivot_row], stranfer_matrix[r] = stranfer_matrix[r], stranfer_matrix[
            pivot_row]
        pivot_cols.append(c)

        # Chuẩn hóa pivot về 1
        pivot_value = stranfer_matrix[r][c]
        # Đưa giá trị pivot về 1 đề dùng khử cột
        stranfer_matrix[r] = [x / pivot_value for x in stranfer_matrix[r]]

        # Khử các dòng còn lại
        for i in range(rows):
            if i != r:
                factor = stranfer_matrix[i][c]
                stranfer_matrix[i] = [stranfer_matrix[i][k] - factor *
                                      stranfer_matrix[r][k] for k in range(cols)]

        r += 1

    return stranfer_matrix, pivot_cols


def rank_and_basis(matrix: list) -> tuple[int, list, list, list]:
    """
     Tính hạng và các cơ sở của ma trận.

    Args:
        matrix : ma trận đầu vào

    Returns:
        tuple gồm:
            - rank        : hạng của ma trận
            - col_basis   : cơ sở không gian cột (lấy từ ma trận gốc)
            - row_basis   : cơ sở không gian dòng (lấy từ RREF)
            - null_basis  : cơ sở không gian nghiệm (null space)
    """
    if not matrix or not matrix[0]:
        return 0, [], [], []

    rows = len(matrix)
    cols = len(matrix[0])

    for r in matrix:
        if len(r) != cols:
            raise ValueError("Ma trận không đúng định dạng")

    rref_matrix, pivot_cols = get_rref(matrix)

    # Hạng ma trận
    rank = len(pivot_cols)

    # Cơ sở không gian cột
    # Lấy cột tại vị trí pivot ở ma trận gốc vì  rref sẽ làm biến đổi không gian cột
    col_basis = []
    for c in pivot_cols:
        col_vector = [matrix[r][c] for r in range(rows)]
        col_basis.append(col_vector)

    # Cơ sở không gian dòng
    # Lấy các dòng khác 0 trong RREF vì rref không làm biến đổi không gian dòng
    row_basis = []
    for r in range(rank):
        row_basis.append(rref_matrix[r])

    # Cơ sở không gian nghiệm
    # Xác định các cột không có pivot -> cột là biến tự do
    free_vars = [j for j in range(cols) if j not in pivot_cols]

    null_basis = []
    for f_var in free_vars:
        vec = [Fraction(0)] * cols

        # Chọn 1 biến tự do bằng 1
        vec[f_var] = Fraction(1)

        # Tính các biến pivot bằng biến tự do
        # Duyệt qua các dòng có pivot
        for i, p_col in enumerate(pivot_cols):
            vec[p_col] = -rref_matrix[i][f_var]

        null_basis.append(vec)

    return rank, col_basis, row_basis, null_basis


def test():
    print("\n--- TEST 1: Ma trận 3x3 đủ hạng ---")
    A = [
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]
    ]
    rank, col_b, row_b, null_b = rank_and_basis(A)
    print(f"Rank: {rank}")
    print("Col basis:", [[str(x) for x in v] for v in col_b])
    print("Row basis:", [[str(x) for x in v] for v in row_b])
    print("Null basis:", null_b)

    print("\n--- TEST 2: Ma trận có null space (hạng < số cột) ---")
    B = [
        [1, 2, 3],
        [2, 4, 6],
        [0, 1, 4]
    ]
    rank, col_b, row_b, null_b = rank_and_basis(B)
    print(f"Rank: {rank}")
    print("Col basis:", [[str(x) for x in v] for v in col_b])
    print("Row basis:", [[str(x) for x in v] for v in row_b])
    print("Null basis:", [[str(x) for x in v] for v in null_b])

    print("\n--- TEST 3: Ma trận không vuông 2x4 ---")
    C = [
        [1, 2, 0, 1],
        [0, 0, 1, 3]
    ]
    rank, col_b, row_b, null_b = rank_and_basis(C)
    print(f"Rank: {rank}")
    print("Col basis:", [[str(x) for x in v] for v in col_b])
    print("Row basis:", [[str(x) for x in v] for v in row_b])
    print("Null basis:", [[str(x) for x in v] for v in null_b])


if __name__ == "__main__":
    test()
