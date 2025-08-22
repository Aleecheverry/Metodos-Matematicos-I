import sympy as sp

# Definimos la variable simbólica x
x = sp.symbols('x')

# Definir el grado n (puedes cambiar este valor)
n = 5  # Por ejemplo, grado 5

# Crear la base canónica de polinomios {1, t, t^2, ..., t^n}
base_canónica = [x**k for k in range(n+1)]

# Definir el peso del producto interno sqrt(1 - x^2)
w = sp.sqrt(1 - x**2)

# Definir el producto interno
def producto_interno(f, g):
    return sp.integrate(f * g * w, (x, -1, 1))

# Definir la proyección
def proyección(f, g):
    # Proyección de f sobre g, usando el producto interno
    denom = producto_interno(g, g)
    return producto_interno(f, g) / denom * g

# Proceso de Gram-Schmidt
base_ortogonal = []
for p in base_canónica:
    # Comenzamos con el polinomio p
    q = p
    # Restamos las proyecciones sobre los polinomios previamente calculados
    for g in base_ortogonal:
        q = q - proyección(q, g)
    # Si el polinomio resultante no es cero, lo añadimos a la base ortogonal
    if q != 0:
        base_ortogonal.append(sp.simplify(q))

# Normalizar la base ortogonal (si se desea base ortonormal)
base_ortonormal = [sp.simplify(q / sp.sqrt(producto_interno(q, q))) for q in base_ortogonal]

# Mostrar los resultados
print("Base ortogonal (sin normalizar):")
for i, pol in enumerate(base_ortogonal):
    print(f"q_{i}(x) = {sp.expand(pol)}")

print("\nBase ortonormal:")
for i, pol in enumerate(base_ortonormal):
    print(f"r_{i}(x) = {sp.expand(pol)}")

# Verificar la ortogonalidad
M = sp.Matrix([[producto_interno(base_ortogonal[i], base_ortogonal[j]) for j in range(n+1)] for i in range(n+1)])
print("\nMatriz de productos internos (debe ser diagonal):")
sp.pprint(M)

# Verificación con los polinomios de Chebyshev U_k(x)
U = [sp.chebyshevu(k, x) for k in range(n+1)]
print("\nComparación con los polinomios de Chebyshev de segunda especie U_k(x):")
for i, u in enumerate(U):
    print(f"U_{i}(x) = {sp.expand(u)}")

# Comparación de la constante de proporcionalidad entre la base ortogonal y los Chebyshev
constantes = []
for i, q in enumerate(base_ortogonal):
    c = sp.simplify(sp.Poly(q, x).LC() / sp.Poly(U[i], x).LC())
    constantes.append(c)
    print(f"Constante de proporcionalidad entre q_{i}(x) y U_{i}(x) = {c}")
