include("firmas.jl")
using FileIO: load
using Images
using Statistics
using Random

## VARIABLES GLOBALES, CAMBIAR CARPETA DE CADA UNO
carpeta_sandstorm = "dataset/sandstorm"
carpeta_lightning = "dataset/lightning"


#cargar todas las imágenes de la carpeta como una matriz tridimensional de canales (RGB) y valores
function cargar_imagenes_carpeta(carpeta)
    imagenes = []
    for archivo in readdir(carpeta)
        if endswith(archivo, ".jpg")
            img = load(joinpath(carpeta, archivo))
            push!(imagenes, channelview(img))
        end
    end
    return imagenes
end

#filtrar imágenes inválidas (con valores NaN o infinitos)
function filtrar_imagenes_invalidas(imagenes)
    return filter(img -> all(.!isnan.(img)) && all(isfinite.(img)), imagenes)
end


# calcular estadísticas de cada imagen (media y desviación estándar)
function calcular_estadisticas(imagen)
    return [mean(imagen), std(imagen)]
end

imagenes_sandstorm = cargar_imagenes_carpeta(carpeta_sandstorm)
imagenes_lightning = cargar_imagenes_carpeta(carpeta_lightning)

imagenes_sandstorm = filtrar_imagenes_invalidas(imagenes_sandstorm)
imagenes_lightning = filtrar_imagenes_invalidas(imagenes_lightning)

estadisticas_sandstorm = calcular_estadisticas.(imagenes_sandstorm)
estadisticas_lightning = calcular_estadisticas.(imagenes_lightning)

[estadisticas_sandstorm; estadisticas_lightning]
#@show typeof.(estadisticas_sandstorm)
#@show typeof.(estadisticas_lightning)
#@show estadisticas_sandstorm[1]

datos_juntos = collect(hcat([estadisticas_sandstorm; estadisticas_lightning]...)')
#ahora mismo estamos marcamos unos como falsos y otros como true para que se llame a crossvalidadation de booleanas
#no se porque manejantolo como strings crossvalidation daba valores de indexes sin sentido y daba OutOfMemoryError() al ejecutar
datos_deseados = vcat(falses(length(estadisticas_sandstorm)), trues(length(estadisticas_lightning)))



# Elegir técnica

# technique = :ANN
technique = :SVC
# technique = :DecisionTreeClassifier
# technique = :KNeighborsClassifier


# Definir los diccionarios de hiperparámetros
dictANN = Dict(
    "topology" => [5],
    "learningRate" => 0.01,
    "validationRatio" => 0.1,
    "numExecutions" => 10,
    "maxEpochs" => 500,
    "maxEpochsVal" => 20
)

dictSVC = Dict("kernel" => "rbf", "C" => 10.0, "gamma" => 0.5, "degree" => 3)
dictTree = Dict("max_depth" => 4)
dictKNN = Dict("n_neighbors" => 3)


indexes = crossvalidation(datos_deseados, 5)  # 5 folds, a elegir
@show maximum(indexes)
@show minimum(indexes)
@show unique(indexes)
@show length(indexes)

# Entrenamiento y evaluación
if technique == :ANN
    resultados = modelCrossValidation(:ANN, dictANN, (datos_juntos, datos_deseados), indexes)
elseif technique == :SVC
    resultados = modelCrossValidation(:SVC, dictSVC, (datos_juntos, datos_deseados), indexes)
elseif technique == :DecisionTreeClassifier
    resultados = modelCrossValidation(:DecisionTreeClassifier, dictTree, (datos_juntos, datos_deseados), indexes)
elseif technique == :KNeighborsClassifier
    resultados = modelCrossValidation(:KNeighborsClassifier, dictKNN, (datos_juntos, datos_deseados), indexes)
end

println("\n--- RESULTADOS ---")
println("Accuracy: ", resultados[1])
println("Error rate: ", resultados[2])
println("Sensitivity: ", resultados[3])
println("Specificity: ", resultados[4])
println("VPP: ", resultados[5])
println("VPN: ", resultados[6])
println("F1 Score: ", resultados[7])
println("Confusion Matrix:\n", resultados[8])

# para ser la primera aproximacion da bueno valores
# he añadido los datos al github y borrado parte de las imagenes de sandstorm para que no estuviera descompensado(habia casi el doble)