import json
from django.shortcuts import render
from .utils import ejecutar_biseccion

PRESETS = [
    {'label': 'Polinómica: x³ - x - 2', 'func': 'x**3 - x - 2', 'a': 1, 'b': 2},
    {'label': 'Trigonométrica: cos(x) - x', 'func': 'cos(x) - x', 'a': 0, 'b': 1},
    {'label': 'Exponencial: e^x - 3x', 'func': 'exp(x) - 3*x', 'a': 0, 'b': 1},
    {'label': 'Logarítmica: ln(x) - x + 2', 'func': 'log(x) - x + 2', 'a': 1, 'b': 4},
]

def index(request):
    context = {'presets': PRESETS}
    if request.method == 'POST':
        try:
            func = request.POST.get('funcion', '').strip()
            a = float(request.POST.get('a'))
            b = float(request.POST.get('b'))
            tol = float(request.POST.get('tolerancia', 0.0001))

            historial, grafica_data, mensaje = ejecutar_biseccion(func, a, b, tol)

            context.update({
                'historial': historial,
                'grafica_json': json.dumps(grafica_data) if grafica_data else None,
                'mensaje': mensaje,
                'raiz_final': historial[-1] if historial else None,
                'funcion': func,
                'a': a,
                'b': b,
                'tolerancia': tol
            })
        except ValueError:
            context['mensaje'] = "Por favor ingresa valores numéricos válidos para los límites y la tolerancia."
        except Exception as e:
            context['mensaje'] = f"Error inesperado: {str(e)}"

    return render(request, 'calculadora/index.html', context)