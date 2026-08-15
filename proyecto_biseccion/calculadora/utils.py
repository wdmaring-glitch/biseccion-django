import sympy as sp
import numpy as np

def ejecutar_biseccion(func_str, a, b, tol=0.0001, max_iter=50):
    x = sp.Symbol('x')
    try:
        # Reemplazar sintaxis común como ^ por **
        func_str_prep = func_str.replace('^', '**')
        expr = sp.sympify(func_str_prep)
        
        # Mapeo de SymPy a NumPy/Math para evaluación eficiente
        f = sp.lambdify(x, expr, modules=['numpy', 'math'])
        
        fa = float(f(a))
        fb = float(f(b))
    except Exception as e:
        return None, None, f"Error al interpretar la expresión: {str(e)}"

    if fa * fb >= 0:
        return None, None, f"No se cumple el Teorema de Bolzano: f({a}) = {fa:.4f} y f({b}) = {fb:.4f} tienen el mismo signo."

    historial = []
    a_curr, b_curr, fa_curr = float(a), float(b), fa
    c_prev = None

    for i in range(1, max_iter + 1):
        c = (a_curr + b_curr) / 2.0
        fc = float(f(c))
        error_abs = abs(b_curr - a_curr) / 2.0
        error_rel = (abs((c - c_prev) / c) * 100) if (c_prev is not None and c != 0) else 100.0
        c_prev = c

        historial.append({
            'iter': i,
            'a': float(round(a_curr, 6)),
            'b': float(round(b_curr, 6)),
            'c': float(round(c, 6)),
            'fc': float(round(fc, 6)),
            'error_abs': float(round(error_abs, 6)),
            'error_rel': float(round(error_rel, 4))
        })

        if abs(fc) < tol or error_abs < tol:
            break

        if fa_curr * fc < 0:
            b_curr = c
        else:
            a_curr = c
            fa_curr = fc

    # Puntos para la gráfica interactiva de la función f(x)
    margin = max(abs(b - a) * 0.5, 1.0)
    x_vals = np.linspace(a - margin, b + margin, 300)
    y_vals = []
    for xv in x_vals:
        try:
            val = float(f(xv))
            y_vals.append(round(val, 5) if np.isfinite(val) else None)
        except Exception:
            y_vals.append(None)

    # Expresión en notación LaTeX para la vista previa
    try:
        latex_str = sp.latex(expr)
    except Exception:
        latex_str = func_str

    grafica_data = {
        'x': [float(round(v, 4)) for v in x_vals],
        'y': y_vals,
        'root_x': float(round(historial[-1]['c'], 6)),
        'root_y': float(round(historial[-1]['fc'], 6)),
        'a_init': float(a),
        'b_init': float(b),
        'latex': latex_str,
        'iters_x': [h['iter'] for h in historial],
        'errors_y': [h['error_abs'] for h in historial]
    }

    return historial, grafica_data, "Proceso completado exitosamente con convergencia."