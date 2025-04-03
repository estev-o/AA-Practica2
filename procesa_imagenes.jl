using FileIO
using Images
using Statistics

## VARIABLES GLOBALES, CAMBIAR CARPETA DE CADA UNO
carpeta_sandstorm = "/home/estevo/AA/Practica2/dataset/sandstorm"
carpeta_lightning = "/home/estevo/AA/Practica2/dataset/lightning"


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


function calcular_estadisticas(imagen)
    return [mean(imagen), std(imagen)]
end

imagenes_sandstorm = cargar_imagenes_carpeta(carpeta_sandstorm)
imagenes_lightning = cargar_imagenes_carpeta(carpeta_lightning)

estadisticas_sandstorm = calcular_estadisticas.(imagenes_sandstorm)
estadisticas_lightning = calcular_estadisticas.(imagenes_lightning)

[estadisticas_sandstorm; estadisticas_lightning]
datos_juntos = collect(hcat([estadisticas_sandstorm; estadisticas_lightning]...)')

datos_deseados = vcat(repeat(["sandstorm"], length(estadisticas_sandstorm)), repeat(["lightning"], length(estadisticas_lightning)))

# todo primero llamar a crossvalidation para sacar los índices y después llamar a modelcrossvalidation para entrenar
