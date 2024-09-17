def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    
    if x0 is None:
        x0 = [0.0] * n
    
    x = x0[:]
    
    for k in range(max_iter):
        x_old = x[:]
        
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Chequear convergencia
        norm = max(abs(x[i] - x_old[i]) for i in range(n))
        if norm < tol:
            print(f'Convergió en {k+1} iteraciones.')
            return x
    
    print('No se alcanzó la convergencia en el número máximo de iteraciones.')
    return x

# Ejemplo de uso
A = [
    [3, -0.1, -0.2],
    [0.1, 7, -0.3],
    [0.3, -0.2, 10]
]

b = [7.85, -19.3, 71.4]

x0 = [0.0] * len(b)

x = gauss_seidel(A, b, x0)
print('Solución:', x)
