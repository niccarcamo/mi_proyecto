from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import joblib  # Para cargar el modelo y el escalador
from .forms import MyPredictionForm  
from django.http import HttpResponse



def index(request):
    return render(request, 'index.html')

def app1(request):
    # Ruta del archivo CSV
    csv_file_path = 'C:/Users/naiko/Desktop/mi_proyecto/Anexo ET_demo_round_traces.csv'
    cs = pd.read_csv(csv_file_path, delimiter=';')

    # Convertir TimeAlive y TravelledDistance a numérico, manejando errores y eliminando puntos decimales incorrectos
    cs['TimeAlive'] = pd.to_numeric(cs['TimeAlive'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    cs['TravelledDistance'] = pd.to_numeric(cs['TravelledDistance'].str.replace('.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')

    # Definir el rango deseado para la normalización
    min_segundos = 0  # Valor mínimo en segundos
    max_segundos = 155  # Valor máximo en segundos

    # Aplicar Min-Max Scaling para normalizar los valores de TimeAlive
    scaler = MinMaxScaler(feature_range=(min_segundos, max_segundos))
    cs['TimeAlive_normalized_seconds'] = scaler.fit_transform(cs[['TimeAlive']])

    # Calcular métricas
    media = cs['TimeAlive'].mean()
    mediana = cs['TimeAlive'].median()
    rango = cs['TimeAlive_normalized_seconds'].max() - cs['TimeAlive_normalized_seconds'].min()
    desviacion_estandar = cs['TimeAlive_normalized_seconds'].std()
    varianza = cs['TimeAlive_normalized_seconds'].var()
    coeficiente_variacion = desviacion_estandar / media

    # Sumar el TimeAlive_normalized_seconds por Team, MatchId y Mapa
    time_alive_sum_by_team_match_map = cs.groupby(['Team', 'MatchId', 'Map'])['TimeAlive_normalized_seconds'].sum()

    # Calcular el promedio de las sumas por Team y Mapa
    time_alive_avg_by_team_map = time_alive_sum_by_team_match_map.groupby(['Map', 'Team']).mean()

    # Filtrar el DataFrame para incluir solo las filas donde MatchWinner no es nulo y es True
    filtered_data = cs[cs['MatchWinner'].notnull() & cs['MatchWinner']]

    # Eliminar duplicados basados en MatchId
    filtered_data_unique = filtered_data.drop_duplicates(subset=['MatchId'])

    # Agrupar por Mapa y Team y contar cuántas veces aparece MatchWinner = True
    grouped_data = filtered_data_unique.groupby(['Map', 'Team']).size()
    
    # Convertir el resultado en un DataFrame para facilitar la manipulación
    grouped_data_df = grouped_data.reset_index(name='Count')

    # Generar gráfico de barras
    fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
    pivot_table = grouped_data_df.pivot_table(index='Map', columns='Team', values='Count', fill_value=0)
    pivot_table.plot(kind='bar', ax=ax_bar, color=['#dd8452', '#4682B4'])
    ax_bar.set_title('Número de partidos ganados por equipo y mapa', fontsize=24, pad=20, fontweight='bold')
    ax_bar.set_xlabel('Mapa', fontsize=14, fontweight='bold')
    ax_bar.set_ylabel('Número de partidos ganados', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title='Equipo', title_fontsize='13', fontsize='12')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    fig_bar.patch.set_facecolor('#D9D9D9')
    ax_bar.set_facecolor('#D9D9D9')
    plt.tight_layout()

    # Convertir el gráfico de barras a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    grafico_barras_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    # Procesar selección de mapa y generar gráfico circular si se ha seleccionado uno
    mapa_seleccionado = request.GET.get('mapa_seleccionado', None)
    grafico_pie_base64 = None

    if mapa_seleccionado and mapa_seleccionado in pivot_table.index:
        filtered_data_mapa = filtered_data_unique[filtered_data_unique['Map'] == mapa_seleccionado]
        grouped_data = filtered_data_mapa.groupby(['Map', 'Team']).size()

        fig_pie, ax_pie = plt.subplots(figsize=(10, 6))
        ax_pie.pie(grouped_data, labels=grouped_data.index.get_level_values('Team'), autopct='%1.1f%%', startangle=140, colors=['#dd8452', '#4682B4'])
        ax_pie.axis('equal')
        ax_pie.set_title(f'Distribución de victorias en {mapa_seleccionado} por equipo', fontsize=24, pad=20, fontweight='bold')
        fig_pie.patch.set_facecolor('#D9D9D9')
        ax_pie.set_facecolor('#D9D9D9')
        plt.tight_layout()

        # Convertir el gráfico circular a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        grafico_pie_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

    context = {
        'grafico_barras': grafico_barras_base64,
        'grafico_pie': grafico_pie_base64,
        'mapa_seleccionado': mapa_seleccionado,
        'mapas': pivot_table.index,
    }

    return render(request, 'app1.html', context)


def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_and_predict(new_data, model, scaler, categorical_columns, X_columns):
    # Implementa la lógica de preprocesamiento y predicción adecuada para tu modelo y datos
    # Asegúrate de adaptar esto según la estructura de tu modelo y datos

    # Ejemplo básico de preprocesamiento y predicción
    # Debes adaptar esta parte a tu modelo específico
    # Por ejemplo:
    # 1. Preprocesamiento de los datos nuevos
    # 2. Transformación de datos categóricos si es necesario
    # 3. Normalización utilizando el escalador cargado
    # 4. Predicción utilizando el modelo cargado
    # 5. Retorno del resultado de la predicción

    # Ejemplo básico:
    new_data_processed = scaler.transform(new_data)
    predictions = model.predict(new_data_processed)

    return predictions


def app2(request):
    if request.method == 'POST':
        # Procesamiento del formulario y predicción
        new_data_input = {
            'Map': request.POST.get('Map'),
            # Agrega aquí el resto de los campos de entrada
        }

        # Cargar modelo y escalador
        model, scaler = load_model_and_scaler('best_rf_model.pkl', 'scaler.pkl')

        # Realizar predicción
        new_data_df = pd.DataFrame(new_data_input, index=[0])
        predictions = preprocess_and_predict(new_data_df, model, scaler, categorical_cols, X_train.columns)

        # Generar gráfico de dispersión
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, rf_preds, color='green', alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Predicciones')
        ax.set_title('Gráfico de Dispersión: Valores Reales vs. Predicciones (Random Forest Regressor)', fontsize=18)

        # Convertir el gráfico a formato base64 para insertarlo en el HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        scatter_plot = f"data:image/png;base64,{image_base64}"

        context = {
            'predictions': predictions,
            'scatter_plot': scatter_plot,
        }
        return render(request, 'app2.html', context)
    else:
        # Renderizar el formulario inicial si es un GET
        map_label_mapping = {0: 'de_dust2', 1: 'de_inferno', 2: 'de_mirage', 3: 'de_nuke'}
        context = {
            'map_label_mapping': map_label_mapping,
        }
        return render(request, 'app2.html', context)

def app3(request):
    # Lógica para la vista de app2
    return render(request, 'app3.html')


