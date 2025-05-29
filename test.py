import numpy as np

def poly_least_squares():
    m = int(input("다항식 차수 m: "))
    n = int(input("데이터 개수 n: "))
    xs = [], ys = []
    for i in range(n):
        x_i, y_i = map(float, input(f"({i+1}) x, y 입력: ").split())
        xs.append(x_i)
        ys.append(y_i)
    xs = np.array(xs)
    y = np.array(ys).reshape(n, 1)
    M = np.column_stack([xs**j for j in range(m+1)])              
    v = np.linalg.inv(M.T @ M) @ M.T @ y

    coeffs = v.flatten()
    for j, cj in enumerate(coeffs):
        print(f"a_{j} = {cj}")
    print("=> 다항식: y =", " + ".join(f"({cj:.4f})*x^{j}" for j, cj in enumerate(coeffs)))

if __name__ == "__main__":
    poly_least_squares()
