import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

def preparar_funcion(func_str):
    """Soporta e, ln, pi, multiplicacion implicita (2x) y potencias (^)."""
    x = sp.Symbol('x')
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    # Diccionario local para constantes y funciones comunes
    local_dict = {
        'e': sp.E,
        'E': sp.E,
        'ln': sp.log,
        'log': sp.log,
        'pi': sp.pi,
        'exp': sp.exp
    }
    
    expr = parse_expr(func_str, transformations=transformations, local_dict=local_dict)
    f = sp.lambdify(x, expr, modules=['numpy', 'math'])
    return x, expr, f

def evaluar_safe(f, val):
    """Evalúa f(val) de forma segura capturando indeterminaciones complejas o de dominio."""
    try:
        res = float(f(val))
        return res if np.isfinite(res) else None
    except Exception:
        return None

# 1. BISECCIÓN (Corregido: parada únicamente por ancho de intervalo)
def metodo_biseccion(f, a, b, tol=0.001, max_iter=50):
    fa, fb = evaluar_safe(f, a), evaluar_safe(f, b)
    if fa is None or fb is None or fa * fb >= 0:
        return None, "No se cumple el Teorema de Bolzano en [a, b] (f(a)*f(b) >= 0)."

    historial = []
    a_c, b_c, fa_c = float(a), float(b), fa
    c_prev = None

    for i in range(1, max_iter + 1):
        c = (a_c + b_c) / 2.0
        fc = evaluar_safe(f, c)
        if fc is None:
            return None, f"Indeterminación al evaluar la función en x = {c}."

        error_abs = abs(b_c - a_c) / 2.0
        error_rel = (abs((c - c_prev) / c) * 100) if (c_prev is not None and c != 0) else 100.0
        c_prev = c

        historial.append({
            'iter': i, 'a': round(a_c, 6), 'b': round(b_c, 6), 'c': round(c, 6),
            'fc': round(fc, 6), 'error_abs': round(error_abs, 6), 'error_rel': round(error_rel, 4)
        })

        # Criterio de parada: solo por semiancho de intervalo o raíz exacta
        if error_abs < tol or fc == 0:
            break

        if fa_c * fc < 0:
            b_c = c
        else:
            a_c = c
            fa_c = fc

    return historial, None

# 2. REGULA FALSI
def metodo_regula_falsi(f, a, b, tol=0.001, max_iter=50):
    fa, fb = evaluar_safe(f, a), evaluar_safe(f, b)
    if fa is None or fb is None or fa * fb >= 0:
        return None, "No se cumple el Teorema de Bolzano en [a, b]."

    historial = []
    a_c, b_c, fa_c, fb_c = float(a), float(b), fa, fb
    c_prev = None

    for i in range(1, max_iter + 1):
        denominador = fb_c - fa_c
        if denominador == 0:
            return None, "División por cero detectada en el denominador de Regula Falsi."

        c = (a_c * fb_c - b_c * fa_c) / denominador
        fc = evaluar_safe(f, c)
        if fc is None:
            return None, f"Indeterminación al evaluar en x = {c}."

        error_abs = abs(c - c_prev) if c_prev is not None else abs(b_c - a_c)
        error_rel = (abs((c - c_prev) / c) * 100) if (c_prev is not None and c != 0) else 100.0
        c_prev = c

        historial.append({
            'iter': i, 'a': round(a_c, 6), 'b': round(b_c, 6), 'c': round(c, 6),
            'fc': round(fc, 6), 'error_abs': round(error_abs, 6), 'error_rel': round(error_rel, 4)
        })

        if abs(fc) < tol or error_abs < tol:
            break

        if fa_c * fc < 0:
            b_c, fb_c = c, fc
        else:
            a_c, fa_c = c, fc

    return historial, None

# 3. NEWTON-RAPHSON
def metodo_newton(x_sym, expr, f, x0, tol=0.001, max_iter=50):
    try:
        df_expr = sp.diff(expr, x_sym)
        df = sp.lambdify(x_sym, df_expr, modules=['numpy', 'math'])
    except Exception as e:
        return None, f"No se pudo derivar la función: {str(e)}"

    historial = []
    x_curr = float(x0)

    for i in range(1, max_iter + 1):
        fx = evaluar_safe(f, x_curr)
        dfx = evaluar_safe(df, x_curr)

        if fx is None or dfx is None or dfx == 0:
            return None, "Derivada igual a cero o fuera de dominio en Newton-Raphson."

        x_next = x_curr - (fx / dfx)
        error_abs = abs(x_next - x_curr)
        error_rel = (abs(error_abs / x_next) * 100) if x_next != 0 else 100.0

        historial.append({
            'iter': i, 'a': round(x_curr, 6), 'b': round(dfx, 6), 'c': round(x_next, 6),
            'fc': round(fx, 6), 'error_abs': round(error_abs, 6), 'error_rel': round(error_rel, 4)
        })

        x_curr = x_next
        if abs(fx) < tol or error_abs < tol:
            break

    return historial, None

# 4. SECANTE
def metodo_secante(f, x0, x1, tol=0.001, max_iter=50):
    historial = []
    x0_c, x1_c = float(x0), float(x1)

    for i in range(1, max_iter + 1):
        f0 = evaluar_safe(f, x0_c)
        f1 = evaluar_safe(f, x1_c)

        if f0 is None or f1 is None or (f1 - f0) == 0:
            return None, "División por cero o punto fuera de dominio en el Método de la Secante."

        x_next = x1_c - f1 * (x1_c - x0_c) / (f1 - f0)
        error_abs = abs(x_next - x1_c)
        error_rel = (abs(error_abs / x_next) * 100) if x_next != 0 else 100.0

        historial.append({
            'iter': i, 'a': round(x0_c, 6), 'b': round(x1_c, 6), 'c': round(x_next, 6),
            'fc': round(f1, 6), 'error_abs': round(error_abs, 6), 'error_rel': round(error_rel, 4)
        })

        x0_c, x1_c = x1_c, x_next
        if abs(f1) < tol or error_abs < tol:
            break

    return historial, None

# BUSCADOR AUTOMÁTICO DE INTERVALOS
def buscar_intervalos_sugeridos(func_str, rango_min=-10, rango_max=10, paso=0.5):
    try:
        _, _, f = preparar_funcion(func_str)
    except Exception:
        return []

    x_vals = np.arange(rango_min, rango_max + paso, paso)
    intervalos = []

    for i in range(len(x_vals) - 1):
        a, b = x_vals[i], x_vals[i + 1]
        fa, fb = evaluar_safe(f, a), evaluar_safe(f, b)
        if fa is not None and fb is not None and fa * fb < 0:
            intervalos.append({'a': round(float(a), 2), 'b': round(float(b), 2)})
            if len(intervalos) >= 4:
                break
    return intervalos

# CALCULADOR PRINCIPAL
def calcular_raiz(func_str, a, b, metodo='biseccion', tol=0.001):
    try:
        x_sym, expr, f = preparar_funcion(func_str)
    except Exception as e:
        return None, None, f"Error al interpretar la expresión: {str(e)}", None

    historial, error = None, None
    x0 = (a + b) / 2.0

    if metodo == 'biseccion':
        historial, error = metodo_biseccion(f, a, b, tol)
    elif metodo == 'regula_falsi':
        historial, error = metodo_regula_falsi(f, a, b, tol)
    elif metodo == 'newton':
        historial, error = metodo_newton(x_sym, expr, f, x0, tol)
    elif metodo == 'secante':
        historial, error = metodo_secante(f, a, b, tol)

    if error or not historial:
        return None, None, error or "Error durante el cálculo.", None

    margin = max(abs(b - a) * 0.5, 1.0)
    x_vals = np.linspace(a - margin, b + margin, 300)
    y_vals = [evaluar_safe(f, xv) for xv in x_vals]

    grafica_data = {
        'x': [float(round(v, 4)) for v in x_vals],
        'y': [round(v, 5) if v is not None else None for v in y_vals],
        'root_x': float(round(historial[-1]['c'], 6)),
        'root_y': float(round(historial[-1]['fc'], 6)),
        'latex': sp.latex(expr),
        'iters_x': [h['iter'] for h in historial],
        'errors_y': [h['error_abs'] for h in historial]
    }

    # COMPARATIVA RESISTENTE A ERRORES
    comparativa = []
    
    def intentar_metodo(nombre, funcion_metodo, *args):
        try:
            res, err = funcion_metodo(*args)
            if res:
                return {
                    'nombre': nombre,
                    'iters': len(res),
                    'raiz': res[-1]['c'],
                    'error_abs': res[-1]['error_abs']
                }
        except Exception:
            pass
        return None

    m1 = intentar_metodo('Bisección', metodo_biseccion, f, a, b, tol)
    m2 = intentar_metodo('Regula Falsi', metodo_regula_falsi, f, a, b, tol)
    m3 = intentar_metodo('Newton-Raphson', metodo_newton, x_sym, expr, f, x0, tol)
    m4 = intentar_metodo('Secante', metodo_secante, f, a, b, tol)

    for m in [m1, m2, m3, m4]:
        if m:
            comparativa.append(m)

    return historial, grafica_data, "Cálculo realizado con éxito.", comparativa