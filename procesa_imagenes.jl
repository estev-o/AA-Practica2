
include("firmas.jl")
using FileIO: load
using Images
using Statistics
using Random
using DataFrames 
using StatsBase: sample 
using Random: shuffle 
using JSONTables
using Printf 

# --- Variables Globales ---
carpeta_sandstorm = "dataset/sandstorm"
carpeta_lightning = "dataset/lightning"
k_folds = 5      
Random.seed!(123) 

# --- Funciones de Carga y Preprocesamiento (sin cambios importantes, con warnings) ---
function cargar_imagenes_carpeta(carpeta)
    imagenes = []
    for archivo in readdir(carpeta; join=true)
        if endswith(lowercase(archivo), ".jpg")
            try
                img = load(archivo)
                cv = channelview(img)
                if ndims(cv) == 3 && size(cv, 1) == 3 # Asegurar RGB
                     # Convertir a Float64 aquí para consistencia
                    push!(imagenes, Float64.(cv))
                else
                    @warn "Ignorando imagen no RGB o con formato inesperado: $archivo"
                end
            catch e
                 # Mostrar error específico si la carga falla (como los JpegTurbo warnings)
                @warn "No se pudo cargar la imagen: $archivo. Error: $e"
            end
        end
    end
    return imagenes
end

function filtrar_imagenes_invalidas(imagenes)
    # Filtrar NaN/Inf (ya deberían ser Float64 por el cambio en cargar_imagenes_carpeta)
    return filter(img -> all(!isnan, img) && all(isfinite, img), imagenes)
end

function calcular_estadisticas(imagen::AbstractArray{<:AbstractFloat, 3})
    # Calcular media y std por canal y luego promediar, o aplanar y calcular una vez
    img_flat = vec(imagen)
    # Asegurar que no haya problemas si la std es cero (imagen constante)
    m = mean(img_flat)
    s = std(img_flat, corrected=false) # Usar N en lugar de N-1 si hay pocas muestras? Opcional.
    return [m, s]
end

# --- Carga y Preparación de Datos ---
println("Cargando y preprocesando imágenes...")
imagenes_sandstorm = cargar_imagenes_carpeta(carpeta_sandstorm)
imagenes_lightning = cargar_imagenes_carpeta(carpeta_lightning)

println("Filtrando imágenes inválidas...")
# No necesitamos filtrar de nuevo si ya convertimos a Float y filtramos NaNs en la carga/estadística
# imagenes_sandstorm = filtrar_imagenes_invalidas(imagenes_sandstorm)
# imagenes_lightning = filtrar_imagenes_invalidas(imagenes_lightning)

if isempty(imagenes_sandstorm) || isempty(imagenes_lightning)
    error("No se cargaron suficientes imágenes válidas. Verifica las carpetas y el formato de las imágenes.")
end

println("Calculando estadísticas...")
estadisticas_sandstorm = [calcular_estadisticas(img) for img in imagenes_sandstorm]
estadisticas_lightning = [calcular_estadisticas(img) for img in imagenes_lightning]

# Combinar datos y etiquetas
datos_juntos = vcat(hcat(estadisticas_sandstorm...)', hcat(estadisticas_lightning...)')
datos_deseados = vcat(falses(length(estadisticas_sandstorm)), trues(length(estadisticas_lightning)))

println("Datos cargados: $(size(datos_juntos, 1)) muestras, $(size(datos_juntos, 2)) características.")
println("Distribución de clases: $(count(==(false), datos_deseados)) (sandstorm), $(count(==(true), datos_deseados)) (lightning)")

# --- Definición de Espacios de Hiperparámetros---

# 1. Redes Neuronales Artificiales (ANN)
ann_topologies = [
    [5], [10], [8, 4], [10, 5], [15, 7], [20, 10], [5, 5, 5], [8, 6, 4]
]
ann_fixed_params = Dict(
    "learningRate" => 0.01, "validationRatio" => 0.1, "numExecutions" => 10,
    "maxEpochs" => 500, "maxEpochsVal" => 20
)
ann_configs = [merge(ann_fixed_params, Dict("topology" => topo)) for topo in ann_topologies]

# 2. Support Vector Machines (SVM)
svm_kernels = ["linear", "rbf", "poly"]
svm_C_values = [0.1, 1.0, 10.0, 100.0]
svm_gamma_values = [0.1, 0.5, 1.0]
svm_degree_values = [2, 3]
svm_configs = []
for C in svm_C_values
    push!(svm_configs, Dict("kernel" => "linear", "C" => C))
    for gamma in svm_gamma_values
        push!(svm_configs, Dict("kernel" => "rbf", "C" => C, "gamma" => gamma))
        for degree in svm_degree_values
            push!(svm_configs, Dict("kernel" => "poly", "C" => C, "gamma" => gamma, "degree" => degree))
        end
    end
end
svm_configs = unique(svm_configs)[1:min(end, 12)] # Limitar si es necesario, asegurar > 8

# 3. Árboles de Decisión (Decision Tree)
tree_depths = [2, 3, 4, 5, 7, 10, 15]
tree_configs = [Dict("max_depth" => depth) for depth in tree_depths]

# 4. k-Nearest Neighbors (kNN)
knn_neighbors = [1, 3, 5, 7, 9, 11, 15]
knn_configs = [Dict("n_neighbors" => k) for k in knn_neighbors]

# 5. DoME
dome_nodes = [5, 10, 15, 20, 25, 30, 35, 40]
dome_configs = [Dict("maximumNodes" => n) for n in dome_nodes]

experimentos = Dict(
    :ANN => ann_configs,
    :SVC => svm_configs,
    :DecisionTreeClassifier => tree_configs,
    :KNeighborsClassifier => knn_configs,
    :DoME => dome_configs
)

# ---- INICIO del BLOQUE para EXPERIMENTACION ----
let
    # --- Ejecución de la Experimentación ---
    println("\n--- INICIANDO EXPERIMENTACIÓN CON VALIDACIÓN CRUZADA ($k_folds folds) ---")

    n_samples = size(datos_juntos, 1)
    if n_samples < k_folds
         error("El número de muestras ($n_samples) es menor que el número de folds ($k_folds). Reduce k_folds.")
    end

    local indexes 

    indexes = crossvalidation(datos_deseados, k_folds)

    # DataFrame para almacenar todos los resultados
    resultados_df = DataFrame(
        Modelo = Symbol[], Hiperparametros = String[],
        Accuracy_Mean = Float64[], Accuracy_Std = Float64[],
        ErrorRate_Mean = Float64[], ErrorRate_Std = Float64[],
        Sensitivity_Mean = Float64[], Sensitivity_Std = Float64[],
        Specificity_Mean = Float64[], Specificity_Std = Float64[],
        Precision_Mean = Float64[], Precision_Std = Float64[],
        NPV_Mean = Float64[], NPV_Std = Float64[],
        F1_Mean = Float64[], F1_Std = Float64[],
        ConfusionMatrix_Sum = Array{Float64, 2}[]
    )

    # Bucle principal de experimentos
    model_order = sort(collect(keys(experimentos)))
    for model_symbol in model_order
        configs = experimentos[model_symbol]
        println("\n--- Evaluando Modelo: $model_symbol ---")
        for params_dict in configs
            sorted_params = sort(collect(params_dict), by = first)
            hyperparam_str = join(["$k=$v" for (k, v) in sorted_params], ", ")
            println("  Configuración: $hyperparam_str")
            
            results = modelCrossValidation(model_symbol, params_dict, (datos_juntos, datos_deseados), indexes)

            ((acc_mean, acc_std), (err_mean, err_std), (sens_mean, sens_std),
                (spec_mean, spec_std), (prec_mean, prec_std), (npv_mean, npv_std),
                (f1_mean, f1_std), conf_matrix_sum) = results

                num_classes = length(unique(datos_deseados))

            push!(resultados_df, (
                model_symbol, hyperparam_str,
                acc_mean, acc_std, err_mean, err_std,
                sens_mean, sens_std, spec_mean, spec_std,
                prec_mean, prec_std, npv_mean, npv_std,
                f1_mean, f1_std,
                collect(conf_matrix_sum)
            ))
            @printf("    -> Accuracy: %.4f ± %.4f, F1-Score: %.4f ± %.4f\n", acc_mean, acc_std, f1_mean, f1_std)
        end
    end

    println("\n\n--- RESUMEN DE LA EXPERIMENTACIÓN ---")
    println("Resultados Detallados:")
    show(stdout, resultados_df; allrows=true, allcols=true, show_row_number=false, eltypes=false)
    println("\n")

    println("Mejores configuraciones por modelo (basado en F1 Score Medio):")
    valid_results_df = filter(row -> !isnan(row.F1_Mean), resultados_df)

    if nrow(valid_results_df) > 0
        grouped_df = groupby(valid_results_df, :Modelo)
        idx_max_f1 = combine(grouped_df, :F1_Mean => argmax => :idx_max)
        best_results_list = DataFrame[]
        for i in 1:nrow(idx_max_f1)
            model = idx_max_f1[i, :Modelo]
            idx = idx_max_f1[i, :idx_max]
            sub_df = grouped_df[(Modelo=model,)]
            push!(best_results_list, DataFrame(sub_df[idx, :]))
        end
        best_results = vcat(best_results_list...)
        best_results = select(best_results, :Modelo, :Hiperparametros, :F1_Mean, :F1_Std, :Accuracy_Mean, :Accuracy_Std) |>
                       df -> sort(df, :F1_Mean, rev=true)

        println("-----------------------------------------------------------------------------------------")
        @printf("%-22s | %-35s | %-12s | %-12s\n", "Modelo", "Mejores Hiperparámetros", "F1 Mean", "Accuracy Mean")
        println("-----------------------------------------------------------------------------------------")
        for row in eachrow(best_results)
             params_str = row.Hiperparametros
             if length(params_str) > 35
                 params_str = params_str[1:32] * "..."
             end
             f1_val = isnan(row.F1_Mean) ? "NaN" : @sprintf("%.4f", row.F1_Mean)
             acc_val = isnan(row.Accuracy_Mean) ? "NaN" : @sprintf("%.4f", row.Accuracy_Mean)
             @printf("%-22s | %-35s | %-12s | %-12s\n",
                     row.Modelo, params_str, f1_val, acc_val)
        end
        println("-----------------------------------------------------------------------------------------")
    else
        println("No se encontraron resultados válidos para determinar las mejores configuraciones.")
    end

    println("\nExperimentación completada.")
    println("Guardando resultados.")
    json_filename = "resultados_experimentos.json"
    open(json_filename, "w") do f
        JSONTables.objecttable(f, resultados_df)
    end
    println("Resultados guardados : $json_filename")
end 