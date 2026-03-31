from fractions import Fraction


def get_partial_pivot(matrix: list, colum: int, start_row: int) -> tuple[int, float]:
    """
    Tìm dòng chứa phần tử có trị lớn nhất tại colum, giúp tránh việc chia cho 0
    Bắt đầu tìm từ start_row trở xuống
    Args:
        matrix : ma trận 
        colum : cột muốn tìm giá trị abs lớn nhất
        start_row : vị trí row bắt đầu duyệt tìm max

    Returns:
        tuple[row, giá trị max]
    """

    n = len(matrix)
    if any(((colum > len(row)) or (colum < 0))for row in matrix):
        raise ValueError("Chỉ số cột không phù hợp")

    max = -1
    idx = 0
    for row in range(start_row, n):
        if abs(matrix[row][colum]) > max:
            max = abs(matrix[row][colum])
            idx = row
    return (idx, max)


def inverse(matrix: list) -> list:
    """
    Dùng Gauss Jordan để tìm ra ma trận nghịch đảo

    Args :
        matrix : ma trận cần tìm nghịch đảo (Ma trận vuông)

    Returns :
        list : ma trận nghịch đảo của matrix
    """

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Ma trận đầu vào buộc phải là ma trận vuông")

    # Tạo matrix nx2n để lưu ma trận nghịch đảo
    augmented = []
    for i in range(n):
        row = [Fraction(x) for x in matrix[i]]
        # Tạo dòng thứ i của ma trận đơn vị
        indentity_row = [Fraction(1) if i == j else Fraction(0)
                         for j in range(n)]
        augmented.append(row + indentity_row)

    for i in range(n):
        # Tim dòng k có abs max
        pivot_row, value = get_partial_pivot(augmented, i, i)

        # Kiểm tra nếu value = 0 => tất cả cột đều = 0 => định thức = 0
        # Sẽ không có ma trận nghịch đảo
        if value == 0:
            raise ValueError("Ma trận có định thức = 0 nên không thể chéo hóa")

        # Hoán đổi giá trị của dòng i với pivot_row nếu khác
        if pivot_row != i:
            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]

        # Chọn giá trị pivot dể khử cột thứ i arg[i][i]
        pivot_value = augmented[i][i]
        # Đưa giá trị pivot về 1 đề dùng khử cột
        for j in range(i, 2*n):
            augmented[i][j] /= pivot_value

        # Duyệt qua các dòng k != i để khử giá trị ở cột i arg[k][i]
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(i, 2*n):
                    augmented[k][j] -= factor * augmented[i][j]

    inverse_matrix = [row[n:] for row in augmented]
    return inverse_matrix


def test():
    # Kịch bản 1: Ma trận vuông khả nghịch (Hợp lệ)
    print("\n--- TEST 1: Ma trận khả nghịch ---")
    A = [
        [1, 2, 5],
        [0, -2, -1],
        [0, -3, -7]
    ]
    try:
        A_inv = inverse(A)
        print("Ma trận nghịch đảo A^-1 là:")
        for row in A_inv:
            # Làm tròn 3 chữ số thập phân cho dễ nhìn
            print([str(val) for val in row])
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")

    # Kịch bản 2: Ma trận có định thức = 0 (Dòng 2 gấp đôi dòng 1)
    print("\n--- TEST 2: Ma trận KHÔNG khả nghịch (det = 0) ---")
    B = [
        [1, 2, 3],
        [2, 4, 6],
        [0, 1, 4]
    ]
    try:
        B_inv = inverse(B)
        print("Kết quả:", B_inv)
    except Exception as e:
        print(f"Bắt lỗi thành công: {e}")

    # Kịch bản 3: Ma trận không vuông
    print("\n--- TEST 3: Ma trận đầu vào không vuông ---")
    C = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    try:
        C_inv = inverse(C)
        print("Kết quả:", C_inv)
    except Exception as e:
        print(f"Bắt lỗi thành công: {e}")


if __name__ == "__main__":
    test()
