import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re # Needed for hyperparameter extraction

# --- Configuración ---
json_filename = 'resultados_cnn_experimentos.json'
N_TOP_CONFIGS = 5 # How many top configurations to show in the detailed plot
sns.set_theme(style="whitegrid")

# --- 1. Carga y Preparación de Datos ---
print(f"Cargando datos desde {json_filename}...")
try:
    with open(json_filename, 'r') as f:
        data = json.load(f)
    df_results = pd.DataFrame(data)
    print("Datos cargados exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo {json_filename}.")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: El archivo {json_filename} no es un JSON válido.")
    exit()
except Exception as e:
    print(f"ERROR inesperado al cargar o procesar el JSON: {e}")
    exit()

print("Limpiando y preparando el DataFrame...")
numeric_cols = ['Accuracy_Mean', 'Accuracy_Std', 'ErrorRate_Mean', 'ErrorRate_Std',
                'Sensitivity_Mean', 'Sensitivity_Std', 'Specificity_Mean', 'Specificity_Std',
                'Precision_Mean', 'Precision_Std', 'NPV_Mean', 'NPV_Std', 'F1_Mean', 'F1_Std']

for col in numeric_cols:
    if col in df_results.columns:
        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

initial_rows = len(df_results)
df_results.dropna(subset=['F1_Mean'], inplace=True)
if len(df_results) < initial_rows:
    print(f"INFO: Se eliminaron {initial_rows - len(df_results)} filas con F1_Mean NaN.")

# --- 2. Extracción de Hiperparámetros Clave ---
print("Extrayendo hiperparámetros desde la descripción de la arquitectura...")

def extract_dropout(arch_string):
    match = re.search(r'dropout=([\d\.]+)', arch_string)
    return float(match.group(1)) if match else 0.0 # Assume 0.0 if not found

def extract_num_conv_layers(arch_string):
    match = re.search(r'conv_filters=(\[.*?\])', arch_string)
    if match:
        filters_str = match.group(1)
        # Count elements in the list string (e.g., "[16, 32]" -> 2 layers)
        # Handles empty list "[]" -> 0 layers, "[16]" -> 1 layer
        if filters_str == '[]':
            return 0
        return filters_str.count(',') + 1
    return 0 # Assume 0 if pattern not found

df_results['DropoutRate'] = df_results['Architecture'].apply(extract_dropout)
df_results['NumConvLayers'] = df_results['Architecture'].apply(extract_num_conv_layers)

print("\nInformación del DataFrame procesado (con hiperparámetros extraídos):")
df_results.info()
# print(df_results[['ConfigID', 'Architecture', 'DropoutRate', 'NumConvLayers', 'F1_Mean']].head())

# --- 3. Generación de Visualizaciones ---

if df_results.empty:
    print("\nERROR: No hay datos válidos para generar visualizaciones después de la limpieza.")
else:
    print("\nGenerando visualizaciones...")

    # --- a) Comparación General F1 por Configuración CNN --- 
    try:
        print("Generando gráfico: F1 por Configuración CNN...")
        df_sorted = df_results.sort_values('F1_Mean', ascending=False).reset_index()

        plt.figure(figsize=(12, 7))
        barplot = sns.barplot(data=df_sorted, x='ConfigID', y='F1_Mean',
                              hue='ConfigID', # Use hue to avoid warning, disable legend
                              order=df_sorted['ConfigID'], palette='viridis',
                              legend=False)

        plt.errorbar(x=range(len(df_sorted)),
                     y=df_sorted['F1_Mean'],
                     yerr=df_sorted['F1_Std'].fillna(0),
                     fmt='none', c='black', capsize=5, label='Std Dev del F1 Score')

        plt.title('F1-Score Medio por Configuración CNN')
        plt.ylabel('F1-Score Medio')
        plt.xlabel('ID de Configuración CNN')
        plt.xticks(ticks=range(len(df_sorted)), labels=df_sorted['ConfigID'], rotation=45, ha='right')
        plt.ylim(bottom=max(0, df_sorted['F1_Mean'].min() - 0.05), top=1.01) # Slightly above 1.0
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('visualizacion_cnn_f1_por_config.png', dpi=300)
        plt.show()
        print("Gráfico 'visualizacion_cnn_f1_por_config.png' guardado.")
    except Exception as e:
        print(f"WARN: No se pudo generar el gráfico de comparación de CNNs: {e}")


    # --- b) F1 Score vs Dropout Rate ---
    try:
        print("Generando gráfico: F1 Score vs Dropout Rate...")
        if 'DropoutRate' in df_results.columns and not df_results['DropoutRate'].isnull().all():
            plt.figure(figsize=(10, 6))
            # Use scatterplot with error bars for individual points
            for i, row in df_results.iterrows():
                 plt.errorbar(x=row['DropoutRate'], y=row['F1_Mean'], yerr=row['F1_Std'],
                              fmt='o', capsize=5, alpha=0.7, label=f"ID {row['ConfigID']}" if i < 10 else None ) # Label first few
            plt.title('F1 Score Medio vs. Tasa de Dropout')
            plt.xlabel('Tasa de Dropout')
            plt.ylabel('F1 Score Medio (con Std Dev)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Config ID") # Legend outside
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
            plt.savefig('visualizacion_cnn_f1_vs_dropout.png', dpi=300)
            plt.show()
            print("Gráfico 'visualizacion_cnn_f1_vs_dropout.png' guardado.")
        else:
             print("WARN: No se encontraron datos válidos de DropoutRate para graficar.")
    except Exception as e:
        print(f"WARN: No se pudo generar el gráfico F1 vs Dropout: {e}")


    # --- c) F1 Score vs Number of Conv Layers ---
    try:
        print("Generando gráfico: F1 Score vs Número de Capas Convolucionales...")
        if 'NumConvLayers' in df_results.columns and not df_results['NumConvLayers'].isnull().all() and df_results['NumConvLayers'].nunique() > 1:
            plt.figure(figsize=(10, 6))
            # Boxplot might be better if multiple configs have the same layer count
            sns.boxplot(data=df_results, x='NumConvLayers', y='F1_Mean', palette='coolwarm')
            sns.stripplot(data=df_results, x='NumConvLayers', y='F1_Mean', color='black', size=4, alpha=0.5) # Overlay points
            plt.title('F1 Score Medio vs. Número de Capas Convolucionales')
            plt.xlabel('Número de Capas Convolucionales')
            plt.ylabel('F1 Score Medio')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('visualizacion_cnn_f1_vs_numlayers.png', dpi=300)
            plt.show()
            print("Gráfico 'visualizacion_cnn_f1_vs_numlayers.png' guardado.")
        elif 'NumConvLayers' in df_results.columns and df_results['NumConvLayers'].nunique() <= 1:
             print("WARN: No hay suficiente variación en el número de capas convolucionales para generar un gráfico útil.")
        else:
             print("WARN: No se encontraron datos válidos de NumConvLayers para graficar.")
    except Exception as e:
        print(f"WARN: No se pudo generar el gráfico F1 vs Num Conv Layers: {e}")


    # --- d) Accuracy vs F1 Score Scatter Plot ---
    try:
        print("Generando gráfico: Accuracy vs F1 Score...")
        plt.figure(figsize=(8, 8))
        scatter = sns.scatterplot(data=df_results, x='Accuracy_Mean', y='F1_Mean', hue='ConfigID', palette='tab10', s=100) # Size points, color by ID
        plt.title('Accuracy Media vs. F1 Score Medio por Configuración')
        plt.xlabel('Accuracy Media')
        plt.ylabel('F1 Score Medio')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axline((0, 0), slope=1, color='grey', linestyle='--', label='Accuracy = F1') # Diagonal line
        plt.xlim(left=max(0, df_results['Accuracy_Mean'].min() - 0.02), right=1.01)
        plt.ylim(bottom=max(0, df_results['F1_Mean'].min() - 0.02), top=1.01)
        # Add annotations for ConfigID
        for i in range(df_results.shape[0]):
            plt.text(df_results['Accuracy_Mean'][i]+0.001, df_results['F1_Mean'][i]+0.001,
                     df_results['ConfigID'][i], fontdict={'size':9})
        plt.legend(title='Config ID / Ref', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('visualizacion_cnn_accuracy_vs_f1.png', dpi=300)
        plt.show()
        print("Gráfico 'visualizacion_cnn_accuracy_vs_f1.png' guardado.")
    except Exception as e:
        print(f"WARN: No se pudo generar el gráfico Accuracy vs F1: {e}")


    # --- e) F1 Mean vs F1 Std (Performance vs Stability) ---
    try:
        print("Generando gráfico: F1 Medio vs F1 Std Dev...")
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(data=df_results, x='F1_Std', y='F1_Mean', hue='ConfigID', palette='tab10', s=100)
        plt.title('Rendimiento (F1 Medio) vs. Estabilidad (F1 Std Dev)')
        plt.xlabel('Desviación Estándar del F1 Score (Menor es más estable)')
        plt.ylabel('F1 Score Medio (Mayor es mejor)')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Annotate points
        for i in range(df_results.shape[0]):
             plt.text(df_results['F1_Std'][i]+0.0005, df_results['F1_Mean'][i],
                      df_results['ConfigID'][i], fontdict={'size':9})
        plt.legend(title='Config ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('visualizacion_cnn_f1_mean_vs_std.png', dpi=300)
        plt.show()
        print("Gráfico 'visualizacion_cnn_f1_mean_vs_std.png' guardado.")
    except Exception as e:
        print(f"WARN: No se pudo generar el gráfico F1 Mean vs Std: {e}")


    # --- f) Detailed Metrics for Top N Configurations ---
    try:
        print(f"Generando gráfico: Métricas Detalladas para Top {N_TOP_CONFIGS} Configuraciones...")
        df_top_n = df_results.nlargest(N_TOP_CONFIGS, 'F1_Mean')

        # Select and rename metrics for clarity in the plot
        metrics_to_plot = {
            'Accuracy_Mean': 'Accuracy',
            'F1_Mean': 'F1 Score',
            'Precision_Mean': 'Precision',
            'Sensitivity_Mean': 'Recall (Sens.)' # Recall is often used
        }
        df_plot_data = df_top_n[['ConfigID'] + list(metrics_to_plot.keys())]
        df_plot_data = df_plot_data.rename(columns=metrics_to_plot)

        # Melt the DataFrame for easy plotting with Seaborn's grouped barplot
        df_melted = pd.melt(df_plot_data, id_vars=['ConfigID'], var_name='MetricType', value_name='Score')

        plt.figure(figsize=(12, 7))
        barplot_detail = sns.barplot(data=df_melted, x='ConfigID', y='Score', hue='MetricType', palette='muted')

        plt.title(f'Métricas Clave para las Top {N_TOP_CONFIGS} Configuraciones CNN (ordenadas por F1 Score)')
        plt.xlabel('ID de Configuración CNN')
        plt.ylabel('Puntuación Media')
        plt.ylim(bottom=max(0, df_melted['Score'].min() - 0.05), top=1.01)
        plt.legend(title='Métrica', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust for legend
        plt.savefig('visualizacion_cnn_top_n_metrics.png', dpi=300)
        plt.show()
        print("Gráfico 'visualizacion_cnn_top_n_metrics.png' guardado.")
    except Exception as e:
        print(f"WARN: No se pudo generar el gráfico de métricas detalladas: {e}")


    # --- g) Matriz de Confusión de la Mejor Configuración CNN --- 
    try:
        print("Generando gráfico: Matriz de Confusión de la Mejor CNN...")
        if df_results['F1_Mean'].isnull().all():
            print("WARN: Todas las F1_Mean son NaN, no se puede encontrar la mejor configuración.")
        else:
            best_overall_idx_num = df_results['F1_Mean'].idxmax()
            best_run = df_results.loc[best_overall_idx_num]

            print(f"\nMejor configuración CNN global encontrada:")
            print(f"  ConfigID: {best_run['ConfigID']}")
            print(f"  Arquitectura: {best_run['Architecture']}")
            print(f"  F1 Medio: {best_run['F1_Mean']:.4f} +/- {best_run['F1_Std']:.4f}")

            conf_matrix_data = best_run['ConfusionMatrix_Sum']
            conf_matrix_np = None

            # Handle different potential formats from JSON (list of lists or flat list)
            if (isinstance(conf_matrix_data, list) and len(conf_matrix_data) == 2 and
                    all(isinstance(row, list) and len(row) == 2 for row in conf_matrix_data)):
                print("INFO: Matriz de confusión detectada como lista de listas 2x2.")
                try:
                     # Assuming [[TN, FP], [FN, TP]] format
                     conf_matrix_np = np.array(conf_matrix_data)
                except Exception as e_conv:
                    print(f"ERROR: No se pudo convertir la lista de listas 2x2 a array numpy: {e_conv}")
            elif isinstance(conf_matrix_data, (list, np.ndarray)) and np.size(conf_matrix_data) == 4:
                 print("WARN: Matriz de confusión detectada como vector 1D(4). Intentando reshape a 2x2 (Asumiendo TN, FP, FN, TP).")
                 try:
                     conf_matrix_np = np.array(conf_matrix_data).reshape((2, 2))
                     print("Reshape a 2x2 exitoso.")
                 except Exception as e_reshape:
                     print(f"ERROR: Falló el reshape de la matriz de confusión 1D: {e_reshape}")
                     conf_matrix_np = None
            elif conf_matrix_data is None or (isinstance(conf_matrix_data, float) and np.isnan(conf_matrix_data)):
                print("WARN: La matriz de confusión para la mejor CNN es NaN o None.")
                conf_matrix_np = None
            else:
                print(f"WARN: Formato inesperado para la matriz de confusión. Datos: {conf_matrix_data}. Tipo: {type(conf_matrix_data)}")
                conf_matrix_np = None

            if conf_matrix_np is not None and conf_matrix_np.shape == (2, 2):
                class_labels = ['Sandstorm (Real)', 'Lightning (Real)'] # Rows: True Class
                predicted_labels = ['Sandstorm (Pred)', 'Lightning (Pred)'] # Columns: Predicted Class

                plt.figure(figsize=(7, 5.5))
                sns.heatmap(conf_matrix_np, annot=True, fmt='.1f', cmap='Blues',
                            xticklabels=predicted_labels, yticklabels=class_labels,
                            annot_kws={"size": 12})
                plt.xlabel('Clase Predicha')
                plt.ylabel('Clase Verdadera')
                title_str = (f'Matriz de Confusión Sumada (Mejor CNN: Config {best_run["ConfigID"]})\n'
                            f'F1 Score Medio={best_run["F1_Mean"]:.3f}')
                plt.title(title_str, fontsize=10)
                plt.tight_layout()
                plt.savefig('visualizacion_mejor_cnn_matriz_confusion.png', dpi=300)
                plt.show()
                print("Gráfico 'visualizacion_mejor_cnn_matriz_confusion.png' guardado.")
            else:
                print("INFO: No se generó el heatmap de la matriz de confusión.")

    except KeyError as e:
        print(f"WARN: No se encontró la columna necesaria '{e}' en los resultados.")
    except Exception as e:
        print(f"WARN: No se pudo generar el heatmap de la matriz de confusión: {e}")

print("\nProceso de visualización completado.")