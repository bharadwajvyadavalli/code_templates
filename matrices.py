
def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def rotate(matrix):
    matrix[:] = transpose(matrix)
    for row in matrix:
        row.reverse()

def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)
        matrix = transpose(matrix)
        matrix.reverse()
    return result

def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    num = 1
    while num <= n * n:
        for i in range(left, right + 1):
            matrix[top][i] = num
            num += 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        for i in range(right, left - 1, -1):
            matrix[bottom][i] = num
            num += 1
        bottom -= 1
        for i in range(bottom, top - 1, -1):
            matrix[i][left] = num
            num += 1
        left += 1
    return matrix

def diagonal_traverse(matrix):
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    result = [[] for _ in range(m + n - 1)]
    for i in range(m):
        for j in range(n):
            if (i + j) % 2 == 0:
                result[i + j].insert(0, matrix[i][j])
            else:
                result[i + j].append(matrix[i][j])
    return [num for diag in result for num in diag]

def set_zeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    rows, cols = set(), set()
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                rows.add(i)
                cols.add(j)
    for i in range(m):
        for j in range(n):
            if i in rows or j in cols:
                matrix[i][j] = 0

def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    while left <= right:
        mid = (left + right) // 2
        num = matrix[mid // n][mid % n]
        if num == target:
            return True
        elif num < target:
            left = mid + 1
        else:
            right = mid - 1
    return False
