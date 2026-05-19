import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import KBinsDiscretizer
import plotly.graph_objs as go
import plotly.utils
import json
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_examen'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_examen'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== AGREGAR ESTO AQUÍ =====
@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)
# ===== FIN =====

def guardar_datos(df):
    session['data'] = df.to_dict('records')
    session['columns'] = df.columns.tolist()

def guardar_datos(df):
    session['data'] = df.to_dict('records')
    session['columns'] = df.columns.tolist()

def cargar_datos():
    if 'data' in session:
        return pd.DataFrame(session['data'])
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        guardar_datos(df)
        return render_template('index.html', mensaje='Archivo cargado correctamente', columnas=df.columns.tolist())
    return redirect(url_for('index'))

@app.route('/series_tiempo')
def series_tiempo():
    df = cargar_datos()
    if df is None:
        return redirect(url_for('index'))
    
    col_fecha = request.args.get('fecha')
    col_valor = request.args.get('valor')
    
    if not col_fecha or not col_valor:
        return render_template('series_tiempo.html', columnas=df.columns.tolist(), error='Selecciona columna de fecha y valor')
    
    # Convertir a datetime
    df[col_fecha] = pd.to_datetime(df[col_fecha], utc=True, errors='coerce')
    df = df.dropna(subset=[col_fecha, col_valor])
    df = df.sort_values(col_fecha)
    valores = df[col_valor].values
    
    if len(valores) < 2:
        return render_template('series_tiempo.html', columnas=df.columns.tolist(), error='Se necesitan al menos 2 datos')
    
    indices = np.arange(len(valores))
    horizonte = 7
    rango_futuro = np.arange(len(valores), len(valores)+horizonte)
    
    # 1. Método Ingenuo
    ingenuo = [valores[-1]] * horizonte
    
    # 2. Método de la Media
    media = [valores.mean()] * horizonte
    
    # 3. Media Móvil (ventana=3)
    ventana = min(3, len(valores))
    movil = []
    ultimos = list(valores[-ventana:])
    for _ in range(horizonte):
        pred = np.mean(ultimos)
        movil.append(pred)
        ultimos = ultimos[1:] + [pred]
    
    # 4. Método de la Deriva
    if len(valores) > 1:
        pendiente = (valores[-1] - valores[0]) / (len(valores)-1)
    else:
        pendiente = 0
    deriva = [valores[-1] + pendiente * (i+1) for i in range(horizonte)]
    
    # 5. Método Ingenuo Estacional (usa el último periodo completo como estacionalidad)
    # Si hay al menos 7 datos, usa la última semana como patrón
    if len(valores) >= 7:
        patron_estacional = valores[-7:]  # Última semana
        estacional = []
        for i in range(horizonte):
            estacional.append(patron_estacional[i % len(patron_estacional)])
    else:
        estacional = [valores[-1]] * horizonte
    
    # Crear gráficos separados
    graphs = []
    
    # Gráfico 1: Método Ingenuo
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=indices, y=valores, mode='lines+markers', name='Histórico', line=dict(color='blue'), marker=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=rango_futuro, y=ingenuo, mode='lines+markers', name='Ingenuo', line=dict(color='red', dash='dash'), marker=dict(color='red', symbol='diamond')))
    fig1.update_layout(title='📊 Método Ingenuo - Predice el último valor', xaxis_title='Tiempo', yaxis_title='Valor', height=400)
    graphs.append(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))
    
    # Gráfico 2: Método de la Media
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=indices, y=valores, mode='lines+markers', name='Histórico', line=dict(color='blue'), marker=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=rango_futuro, y=media, mode='lines+markers', name='Media', line=dict(color='orange', dash='dash'), marker=dict(color='orange', symbol='cross')))
    fig2.update_layout(title='📊 Método de la Media - Usa el promedio histórico', xaxis_title='Tiempo', yaxis_title='Valor', height=400)
    graphs.append(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))
    
    # Gráfico 3: Media Móvil
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=indices, y=valores, mode='lines+markers', name='Histórico', line=dict(color='blue'), marker=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=rango_futuro, y=movil, mode='lines+markers', name='Media Móvil', line=dict(color='green', dash='dash'), marker=dict(color='green', symbol='square')))
    fig3.update_layout(title='📊 Método de Media Móvil - Promedia los últimos valores', xaxis_title='Tiempo', yaxis_title='Valor', height=400)
    graphs.append(json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder))
    
    # Gráfico 4: Método de la Deriva
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=indices, y=valores, mode='lines+markers', name='Histórico', line=dict(color='blue'), marker=dict(color='blue')))
    fig4.add_trace(go.Scatter(x=rango_futuro, y=deriva, mode='lines+markers', name='Deriva', line=dict(color='purple', dash='dash'), marker=dict(color='purple', symbol='triangle-up')))
    fig4.update_layout(title='📊 Método de la Deriva - Proyección lineal', xaxis_title='Tiempo', yaxis_title='Valor', height=400)
    graphs.append(json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder))
    
    # Gráfico 5: Método Ingenuo Estacional
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=indices, y=valores, mode='lines+markers', name='Histórico', line=dict(color='blue'), marker=dict(color='blue')))
    fig5.add_trace(go.Scatter(x=rango_futuro, y=estacional, mode='lines+markers', name='Ingenuo Estacional', line=dict(color='brown', dash='dash'), marker=dict(color='brown', symbol='star')))
    fig5.update_layout(title='📊 Método Ingenuo Estacional - Repite el patrón estacional', xaxis_title='Tiempo', yaxis_title='Valor', height=400)
    graphs.append(json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder))
    
    return render_template('series_tiempo.html', graphs=graphs, columnas=df.columns.tolist())

@app.route('/mejor_modelo')
def mejor_modelo():
    df = cargar_datos()
    if df is None:
        return redirect(url_for('index'))
    
    col_fecha = request.args.get('fecha')
    col_valor = request.args.get('valor')
    horizonte_unidad = request.args.get('horizonte_unidad', 'dias')
    
    if not col_fecha or not col_valor:
        return render_template('mejor_modelo.html', columnas=df.columns.tolist(), error='Selecciona columnas')
    
    # Convertir fecha si es posible
    try:
        if col_fecha in df.columns:
            df[col_fecha] = pd.to_datetime(df[col_fecha], utc=True, errors='coerce')
            df = df.dropna(subset=[col_fecha])
            df = df.sort_values(col_fecha)
    except:
        pass
    
    # Verificar que la columna de valor sea numérica
    if not np.issubdtype(df[col_valor].dtype, np.number):
        return render_template('mejor_modelo.html', columnas=df.columns.tolist(), 
                              error=f'La columna "{col_valor}" no es numérica. Selecciona una columna con números.')
    
    valores = df[col_valor].values
    
    # Determinar horizonte
    if horizonte_unidad == 'dias':
        horizonte = 7
    elif horizonte_unidad == 'semanas':
        horizonte = 4
    elif horizonte_unidad == 'meses':
        horizonte = 3
    elif horizonte_unidad == 'trimestres':
        horizonte = 2
    else:
        horizonte = 1
    
    if len(valores) <= horizonte:
        return render_template('mejor_modelo.html', columnas=df.columns.tolist(), 
                              error=f'No hay suficientes datos. Se necesitan al menos {horizonte+1} datos. Actualmente hay {len(valores)}.')
    
    train = valores[:-horizonte]
    test = valores[-horizonte:]
    
    errores = {}
    
    # Método Ingenuo
    pred_ing = [train[-1]] * horizonte
    errores['Ingenuo'] = float(np.mean((test - pred_ing)**2))
    
    # Método de la Media
    pred_media = [train.mean()] * horizonte
    errores['Media'] = float(np.mean((test - pred_media)**2))
    
    # Media Móvil (ventana=3)
    ventana = min(3, len(train))
    if ventana > 0:
        pred_movil = []
        ultimos = list(train[-ventana:])
        for _ in range(horizonte):
            p = np.mean(ultimos)
            pred_movil.append(p)
            ultimos = ultimos[1:] + [p]
        errores['Media Móvil'] = float(np.mean((test - pred_movil)**2))
    else:
        errores['Media Móvil'] = float('inf')
    
    # Método de la Deriva
    if len(train) > 1:
        pendiente = (train[-1] - train[0]) / (len(train)-1)
        pred_deriva = [train[-1] + pendiente * (i+1) for i in range(horizonte)]
        errores['Deriva'] = float(np.mean((test - pred_deriva)**2))
    else:
        errores['Deriva'] = float('inf')
    
    # Método Ingenuo Estacional (si hay suficientes datos)
    if len(train) >= 7:
        patron = train[-7:]
        pred_estacional = []
        for i in range(horizonte):
            pred_estacional.append(patron[i % len(patron)])
        errores['Ingenuo Estacional'] = float(np.mean((test - pred_estacional)**2))
    
    mejor = min(errores, key=errores.get)
    
    return render_template('mejor_modelo.html', errores=errores, mejor=mejor, columnas=df.columns.tolist())

@app.route('/regresion_multiple')
def regresion_multiple():
    df = cargar_datos()
    if df is None:
        return redirect(url_for('index'))
    
    cols = df.columns.tolist()
    target = request.args.get('target')
    features = request.args.getlist('features')
    
    if not target or not features:
        return render_template('regresion_multiple.html', columnas=cols, error='Selecciona variable objetivo y características')
    
    # Guardar en sesión para usarlo en predicción
    session['target'] = target
    session['features'] = features
    
    # Filtrar solo columnas numéricas
    X = df[features].select_dtypes(include=[np.number])
    y = df[target].values
    
    # Verificar que la variable objetivo sea numérica
    if not np.issubdtype(y.dtype, np.number):
        return render_template('regresion_multiple.html', columnas=cols, error='La variable objetivo debe ser numérica')
    
    if len(X.columns) == 0:
        return render_template('regresion_multiple.html', columnas=cols, error='Las características deben ser numéricas')
    
    if len(X) < len(features) + 1:
        return render_template('regresion_multiple.html', columnas=cols, error=f'Se necesitan al menos {len(features)+1} datos. Solo hay {len(X)}.')
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Guardar el modelo entrenado en sesión (coeficientes)
    session['coeficientes'] = modelo.coef_.tolist()
    session['intercepto'] = float(modelo.intercept_)
    session['r2'] = float(modelo.score(X, y))
    
    coeficientes = dict(zip(features, modelo.coef_))
    intercepto = modelo.intercept_
    
    ecuacion = f"{target} = {intercepto:.2f}"
    for var, coef in coeficientes.items():
        signo = "+" if coef >= 0 else "-"
        ecuacion += f" {signo} ({abs(coef):.4f} * {var})"
    
    r2 = modelo.score(X, y)
    
    return render_template('regresion_multiple.html', ecuacion=ecuacion, r2=r2, 
                          coeficientes=coeficientes, intercepto=intercepto, columnas=cols)

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    df = cargar_datos()
    if df is None:
        return redirect(url_for('index'))
    
    # Obtener target y features de la sesión (guardados en regresión múltiple)
    target = session.get('target')
    features = session.get('features')
    
    if not target or not features:
        return render_template('prediccion.html', columnas=df.columns.tolist(), 
                              error='Primero debes calcular un modelo en Regresión Múltiple')
    
    if request.method == 'POST':
        valores = {}
        for f in features:
            if f in request.form and request.form[f]:
                valores[f] = float(request.form[f])
            else:
                valores[f] = 0
        
        # Seleccionar solo columnas numéricas
        X = df[features].select_dtypes(include=[np.number])
        y = df[target].values
        
        if len(X.columns) == 0:
            return render_template('prediccion.html', columnas=df.columns.tolist(), 
                                  error='Las variables deben ser numéricas', target=target, features=features)
        
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Crear array con los valores en el orden correcto
        X_pred = [valores[f] for f in features if f in X.columns]
        prediccion_valor = modelo.predict([X_pred])[0]
        
        return render_template('prediccion.html', columnas=df.columns.tolist(), 
                              prediccion=prediccion_valor, valores=valores, 
                              features=features, target=target)
    
    return render_template('prediccion.html', columnas=df.columns.tolist(), 
                          features=features, target=target)

@app.route('/matriz_confusion')
def matriz_confusion():
    df = cargar_datos()
    if df is None:
        return redirect(url_for('index'))
    
    col_real = request.args.get('real')
    col_predicha = request.args.get('predicha')
    
    if not col_real or not col_predicha:
        return render_template('matriz_confusion.html', columnas=df.columns.tolist(), error='Selecciona columnas real y predicha')
    
    y_real = df[col_real].values
    y_pred = df[col_predicha].values
    
    # Convertir a categóricas si son numéricas
    if y_real.dtype.kind in 'if' and y_pred.dtype.kind in 'if':
        from sklearn.preprocessing import KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        y_real = discretizer.fit_transform(y_real.reshape(-1,1)).ravel()
        y_pred = discretizer.transform(y_pred.reshape(-1,1)).ravel()
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_real, y_pred)
    
    # Convertir cm a lista de listas para evitar problemas con numpy
    cm_list = cm.tolist()
    
    # Calcular métricas
    accuracy = float(accuracy_score(y_real, y_pred))
    precision = float(precision_score(y_real, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_real, y_pred, average='weighted', zero_division=0))
    
    print("=== DEBUG MATRIZ CONFUSION ===")
    print(f"Columna real: {col_real}")
    print(f"Columna predicha: {col_predicha}")
    print(f"Matriz: {cm_list}")
    print(f"Accuracy: {accuracy}")
    print("==============================")
    
    return render_template('matriz_confusion.html', 
                          cm=cm_list, 
                          tiene_datos=True,
                          precision=precision, 
                          recall=recall, 
                          accuracy=accuracy, 
                          columnas=df.columns.tolist())

if __name__ == '__main__':
    app.run(debug=True)