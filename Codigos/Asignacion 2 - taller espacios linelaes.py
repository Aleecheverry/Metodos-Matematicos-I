# chebyshev_gs.py
# Genera la base ortogonal en P^n con peso w(x)=sqrt(1-x^2) (Chebyshev U_n)
# e imprime p_n(x), U_n(x), factor de proporcionalidad y norma.
# Incluye verificación de ortogonalidad y exportación a CSV.
#
# Requisitos: sympy  (pip install sympy)

import argparse                                    # [I/O] parsear argumentos CLI (1 vez por ejecución)
import csv                                         # [I/O] exportar CSV (si se pide)
import sys
import sympy as sp
from sympy.functions.combinatorial.numbers import binomial

# -----------------------------
# Config simbólica
# -----------------------------
x = sp.symbols('x', real=True)                     # [Álgebra simbólica] símbolo de variable (1 vez)

# -----------------------------
# Momentos cerrados: m_k = ∫_{-1}^{1} x^k sqrt(1-x^2) dx
#   m_{2r} = π/(2(r+1)) * C(2r, r) / 4^r,  m_{2r+1}=0
# -----------------------------
def moment(k: int) -> sp.Expr:
    """Devuelve el momento m_k = ∫_{-1}^{1} x^k sqrt(1-x^2) dx (forma cerrada)."""
    if k % 2 == 1:                                  # [Propiedad] momentos impares son cero (siempre)
        return sp.Integer(0)
    r = k // 2
    return sp.pi * binomial(2*r, r) / (2 * (r + 1) * 4**r)  # [Fórmula] exacta (cada vez que se pida)

# -----------------------------
# Producto interno para polinomios vía momentos
# -----------------------------
def ip_poly(f: sp.Expr, g: sp.Expr) -> sp.Expr:
    """<f|g> = ∫_{-1}^{1} f(x) g(x) sqrt(1-x^2) dx usando momentos cerrados."""
    f = sp.expand(f)
    g = sp.expand(g)
    F = sp.Poly(f, x)
    G = sp.Poly(g, x)
    s = sp.Integer(0)
    # [Bucle] sumar coeficientes * momento (frecuencia: grados de f por grados de g)
    for i, a in enumerate(F.all_coeffs()[::-1]):    # coef de x^i
        if a == 0:
            continue
        for j, b in enumerate(G.all_coeffs()[::-1]):# coef de x^j
            if b == 0:
                continue
            s += a * b * moment(i + j)
    return sp.simplify(sp.together(sp.expand(s)))   # [Simbólico] simplificar (moderado)

# -----------------------------
# Gram–Schmidt con peso w(x)=sqrt(1-x^2)
# -----------------------------
def gram_schmidt_weighted(n: int):
    """Devuelve [p_0,...,p_n] ortogonales en {1,x,...,x^n} con el producto interno dado."""
    basis = [x**k for k in range(n + 1)]
    ortho = []
    for p in basis:                                  # [Bucle] n+1 iteraciones
        q = sp.expand(p)                             # [Álgebra] expandir monomio (barato)
        for v in ortho:                              # [Bucle] proyecciones contra polinomios previos
            q = sp.simplify(q - ip_poly(q, v) / ip_poly(v, v) * v)
        ortho.append(sp.expand(q))
    return ortho

# -----------------------------
# Utilidades de impresión/CSV
# -----------------------------
def poly_to_str(expr: sp.Expr) -> str:
    """Representación legible de un polinomio."""
    return sp.sstr(sp.expand(expr))

def export_csv(rows, path: str):
    """Exporta resultados a CSV plano (sin dependencias extra)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "p_n(x)", "U_n(x)", "c_n (p_n=c_n*U_n)", "<p_n|p_n>"])
        for r in rows:
            w.writerow([r["n"], r["pn"], r["Un"], r["c"], r["norm"]])

# -----------------------------
# Programa principal
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Gram–Schmidt en P^n con peso sqrt(1-x^2) (Chebyshev U_n).")
    ap.add_argument("--n", type=int, default=8, help="grado máximo n (por defecto 8)")
    ap.add_argument("--csv", type=str, default="", help="ruta para exportar CSV (opcional)")
    ap.add_argument("--show-gram", action="store_true", help="muestra la matriz de Gram (opcional)")
    args = ap.parse_args()

    n = args.n
    if n < 0:
        print("n debe ser ≥ 0", file=sys.stderr)
        sys.exit(1)

    print(f"\n== Base ortogonal en P^{n} con w(x)=sqrt(1-x^2) ==\n")

    # [Cálculo] Gram–Schmidt (n+1 polinomios)
    P = gram_schmidt_weighted(n)

    rows = []
    for k, pk in enumerate(P):
        Uk = sp.chebyshevu(k, x)                     # [Referencia] Chebyshev U_k (cada k)
        # factor c_k tal que p_k = c_k * U_k (compara coef. líder)
        c_k = sp.simplify(sp.expand(pk).as_poly(x).LC() / sp.expand(Uk).as_poly(x).LC())
        norm = ip_poly(pk, pk)                        # [Norma] <p_k|p_k>
        rows.append({
            "n": k,
            "pn": poly_to_str(pk),
            "Un": poly_to_str(Uk),
            "c": sp.sstr(c_k),
            "norm": sp.sstr(norm)
        })

    # [Salida] imprime resultados
    for r in rows:
        print(f"n={r['n']}")
        print(f"  p_{r['n']}(x) = {r['pn']}")
        print(f"  U_{r['n']}(x) = {r['Un']}")
        print(f"  c_{r['n']} tal que p_{r['n']} = c_{r['n']} * U_{r['n']}  =>  c_{r['n']} = {r['c']}")
        print(f"  <p_{r['n']}|p_{r['n']}> = {r['norm']}\n")

    # [Chequeo] ortogonalidad -> matriz Gram debe ser diagonal
    if args.show_gram:
        print("Matriz de Gram (debe ser diagonal):")
        G = sp.Matrix([[ip_poly(P[i], P[j]) for j in range(n + 1)] for i in range(n + 1)])
        sp.pprint(G)

    # [Exportación] CSV (si se pidió)
    if args.csv:
        export_csv(rows, args.csv)
        print(f"\nCSV guardado en: {args.csv}")

    # [Nota] Relación esperada: p_k(x) = 2^{-k} * U_k(x) y ||p_k||^2 = π/(2*4^k)

if __name__ == "__main__":
    main()
