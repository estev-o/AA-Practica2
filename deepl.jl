# --- Dependencies ---
using Flux
using FileIO: load
using Flux.Losses
using Flux: onehotbatch, onecold, @epochs, params 
using Images                  
using ImageMagick            
using Statistics
using Random
using DataFrames
using Printf
using JSONTables
using MLJBase: categorical, mode 
include("firmas.jl")            


# --- Configuration Constants ---
const TARGET_IMG_SIZE = (128, 128) # Tamaño fijo para la entrada de la CNN (Ancho, Alto)
const N_CHANNELS = 3               # Imágenes RGB
const FOLDER_SANDSTORM = "dataset/sandstorm"
const FOLDER_LIGHTNING = "dataset/lightning"
const CLASS_LABELS = ["sandstorm", "lightning"] 
const K_FOLDS = 5 # Número de particiones para validación cruzada
const RANDOM_SEED = 123

# Training Hyperparameters 
const LEARNING_RATE = 0.001
const NUM_EPOCHS = 40 
const BATCH_SIZE = 16 

Random.seed!(RANDOM_SEED)

# --- Data Loading and Preprocessing ---
function load_and_preprocess_images(folder_path::String, target_size::Tuple{Int, Int})
    """
    Carga, redimensiona y convierte imágenes de una carpeta a matrices Float32 RGB.
    """
    images_processed = []
    println("Processing folder: $folder_path")
    files = readdir(folder_path; join=true)
    filter!(f -> lowercase(f) |> x -> endswith(x, ".jpg") || endswith(x, ".jpeg") || endswith(x, ".png") || endswith(x, ".bmp"), files) 
    count = 0
    skipped = 0
    # Itera sobre cada archivo encontrado en la carpeta
    for file_path in files
        try
            # Carga, convierte a RGB Float32 y redimensiona la imagen
            img = load(file_path)
            img_rgb_f32 = convert.(RGB{Float32}, img) 

            img_resized = imresize(img_rgb_f32, target_size)

            # Añade la imagen procesada a la lista
            push!(images_processed, img_resized)
            count += 1
        catch e
            @warn "Could not load or process image: $file_path. Error: $e"
            skipped += 1
        end
    end
    println("Successfully loaded $count images. Skipped $skipped images.")
    if count == 0 && skipped > 0
         @warn "No images were successfully loaded from $folder_path. Check image formats and paths."
    elseif count == 0 && skipped == 0
         @warn "No image files found in $folder_path."
    end
    return images_processed
end


function convert_to_hwcn(images::Vector)
"""
Convierte una matriz de imágenes HW (Alto x Ancho)
en una matriz HWCN (Alto, Ancho, Canales, N) Float32 para Flux.
"""
    if isempty(images)
        return Array{Float32, 4}(undef, TARGET_IMG_SIZE[2], TARGET_IMG_SIZE[1], N_CHANNELS, 0)
    end
    @assert all(size(img) == (TARGET_IMG_SIZE[2], TARGET_IMG_SIZE[1]) for img in images) "Not all images have the target size $(TARGET_IMG_SIZE)"

    num_patterns = length(images)
    # Prepara un array 4D para almacenar los datos en formato HWCN
    data_array = Array{Float32, 4}(undef, TARGET_IMG_SIZE[2], TARGET_IMG_SIZE[1], N_CHANNELS, num_patterns)

    # Itera sobre cada imagen para convertirla y añadirla al array final
    for i in 1:num_patterns
        img_ch_view = channelview(images[i]) # Obtiene vista por canales (Channels, Height, Width)
        # Reordena las dimensiones a (Height, Width, Channels) y asigna al array
        data_array[:, :, :, i] .= permutedims(img_ch_view, (2, 3, 1))
    end
    return data_array
end

# --- Load Data ---
# Carga y preprocesa las imágenes de cada clase
println("Loading Sandstorm images...")
imgs_sandstorm = load_and_preprocess_images(FOLDER_SANDSTORM, (TARGET_IMG_SIZE[1], TARGET_IMG_SIZE[2])) 
println("Loading Lightning images...")
imgs_lightning = load_and_preprocess_images(FOLDER_LIGHTNING, (TARGET_IMG_SIZE[1], TARGET_IMG_SIZE[2]))

if isempty(imgs_sandstorm) || isempty(imgs_lightning)
    error("Failed to load sufficient images from one or both classes. Check paths and image files.")
end

# Combina las imágenes y crea las etiquetas correspondientes
all_images = vcat(imgs_sandstorm, imgs_lightning)
all_labels = vcat(
    fill(CLASS_LABELS[1], length(imgs_sandstorm)),
    fill(CLASS_LABELS[2], length(imgs_lightning))
)

println("Total images loaded: $(length(all_images))")
println("Class distribution: $(count(==(CLASS_LABELS[1]), all_labels)) $(CLASS_LABELS[1]), $(count(==(CLASS_LABELS[2]), all_labels)) $(CLASS_LABELS[2])")


# --- CNN Architectures Definition ---
# Función para crear un modelo CNN basado en una configuración dada
function create_cnn_model(config::Dict)
    # Extrae parámetros de la configuración
    filter_sizes = get(config, "conv_filters", [16, 32, 32]) 
    kernel_size = get(config, "kernel_size", (3, 3))
    dense_neurons = get(config, "dense_neurons", 64)
    use_dropout = get(config, "dropout", 0.0) > 0.0
    dropout_rate = get(config, "dropout", 0.0)
    activation = relu 

    in_channels = N_CHANNELS
    
    layers = []
    
    current_height = TARGET_IMG_SIZE[2]
    current_width = TARGET_IMG_SIZE[1]

    # Construye las capas convolucionales y de pooling
    for (i, out_channels) in enumerate(filter_sizes)
        push!(layers, Conv(kernel_size, in_channels => out_channels, activation; pad=SamePad()))# Capa convolucional
        push!(layers, MaxPool((2, 2)))# Capa de Max Pooling
        # Actualiza dimensiones tras pooling
        current_height = ceil(Int, current_height / 2) 
        current_width = ceil(Int, current_width / 2)
        in_channels = out_channels  # Actualiza canales de entrada para la siguiente capa
    end

    push!(layers, Flux.flatten)

    flattened_size = current_height * current_width * in_channels
    
    
    # Construye las capas densas (totalmente conectadas)
    push!(layers, Dense(flattened_size, dense_neurons, activation))
    if use_dropout
        push!(layers, Dropout(dropout_rate))
    end
    push!(layers, Dense(dense_neurons, length(CLASS_LABELS))) 

    return Chain(layers...)
end

# First 10 architectures
#==
cnn_architectures = [
    Dict("id" => 1, "conv_filters" => [16, 32], "dense_neurons" => 32, "dropout" => 0.0),
    Dict("id" => 2, "conv_filters" => [16, 32], "dense_neurons" => 64, "dropout" => 0.0),
    Dict("id" => 3, "conv_filters" => [8, 16, 32], "dense_neurons" => 64, "dropout" => 0.0),
    Dict("id" => 4, "conv_filters" => [16, 32, 64], "dense_neurons" => 64, "dropout" => 0.0),
    Dict("id" => 5, "conv_filters" => [16, 32, 32], "dense_neurons" => 128, "dropout" => 0.0),
    Dict("id" => 6, "conv_filters" => [16, 32], "dense_neurons" => 64, "dropout" => 0.25),
    Dict("id" => 7, "conv_filters" => [16, 32, 64], "dense_neurons" => 64, "dropout" => 0.25),
    Dict("id" => 8, "conv_filters" => [16, 32, 64], "dense_neurons" => 128, "dropout" => 0.25),
    Dict("id" => 9, "conv_filters" => [32, 64], "dense_neurons" => 128, "dropout" => 0.3),
    Dict("id" => 10, "conv_filters" => [32, 64, 128], "dense_neurons" => 128, "dropout" => 0.3),
    # Add more variations if needed (e.g., different kernel sizes, activations)
]==#

cnn_architectures = [
    Dict("id" => 1,  "conv_filters" => [16, 32],       "dense_neurons" => 32,  "dropout" => 0.0,  "batchnorm" => false), # Original Config 1 
    Dict("id" => 3,  "conv_filters" => [8, 16, 32],    "dense_neurons" => 64,  "dropout" => 0.0,  "batchnorm" => false), # Original Config 3 
    Dict("id" => 5,  "conv_filters" => [16, 32, 32],   "dense_neurons" => 128, "dropout" => 0.0,  "batchnorm" => false), # Original Config 5 
    Dict("id" => 10, "conv_filters" => [32, 64, 128],  "dense_neurons" => 128, "dropout" => 0.3,  "batchnorm" => false), # Original Config 10 (Best)

    Dict("id" => 11, "conv_filters" => [32, 64, 128],  "dense_neurons" => 128, "dropout" => 0.2,  "batchnorm" => false), 
    Dict("id" => 12, "conv_filters" => [32, 64, 128],  "dense_neurons" => 128, "dropout" => 0.4,  "batchnorm" => false), 

    Dict("id" => 13, "conv_filters" => [16, 32, 64],   "dense_neurons" => 128, "dropout" => 0.15, "batchnorm" => false),

    Dict("id" => 14, "conv_filters" => [16, 32, 32],   "dense_neurons" => 128, "dropout" => 0.0,  "batchnorm" => true), 
    Dict("id" => 15, "conv_filters" => [32, 64, 128],  "dense_neurons" => 128, "dropout" => 0.3,  "batchnorm" => true), 

    Dict("id" => 16, "conv_filters" => [32, 64],       "dense_neurons" => 256, "dropout" => 0.3,  "batchnorm" => false), 
    Dict("id" => 17, "conv_filters" => [16, 32, 64, 64],"dense_neurons" => 128, "dropout" => 0.25, "batchnorm" => false), 
    Dict("id" => 18, "conv_filters" => [32, 64, 64, 128],"dense_neurons" => 128, "dropout" => 0.3, "batchnorm" => true),  

]

# --- Cross-Validation Function for CNN ---


function cnn_cross_validation(cnn_config::Dict,
                              all_images_original::Vector, 
                              all_labels::Vector{String},
                              cv_indices::Vector{Int})
"""
Realiza una validación cruzada K-fold para una arquitectura CNN determinada.
"""
    num_folds = maximum(cv_indices)
    @assert minimum(cv_indices) == 1 "Cross-validation indices should start from 1."

    fold_results = [] 

    num_classes = length(CLASS_LABELS)
    confusion_matrix_total = zeros(Float64, num_classes, num_classes)

    println("\n--- Evaluating CNN Config ID: $(cnn_config["id"]) ---")
    config_str = join(["$k=$v" for (k, v) in cnn_config if k != "id"], ", ")
    println("  Config: $config_str")

    for k in 1:num_folds
        println("  Fold $k/$num_folds")

        test_idx = findall(cv_indices .== k)
        train_idx = findall(cv_indices .!= k)

        train_imgs_fold = all_images_original[train_idx]
        test_imgs_fold = all_images_original[test_idx]
        train_labels_fold = all_labels[train_idx]
        test_labels_fold = all_labels[test_idx]

        train_x = convert_to_hwcn(train_imgs_fold)
        test_x = convert_to_hwcn(test_imgs_fold)

        train_y = onehotbatch(train_labels_fold, CLASS_LABELS)
        test_y = onehotbatch(test_labels_fold, CLASS_LABELS) 

        train_loader = Flux.DataLoader((train_x, train_y), batchsize=BATCH_SIZE, shuffle=true, partial=false)

        model = create_cnn_model(cnn_config)
        
        loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), y)

        optimizer = Adam(LEARNING_RATE)
        opt_state = Flux.setup(optimizer, model)

        println("    Training for $NUM_EPOCHS epochs...")
        for epoch in 1:NUM_EPOCHS
            epoch_loss = 0.0
            count = 0
            for (batch_x, batch_y) in train_loader
                batch_loss, grads = Flux.withgradient(m -> loss_fn(m, batch_x, batch_y), model)
                Flux.update!(opt_state, model, grads[1])    
                epoch_loss += batch_loss * size(batch_x, 4)
                count += size(batch_x, 4)
            end
            avg_epoch_loss = epoch_loss / count
            if epoch % 5 == 0 || epoch == 1 || epoch == NUM_EPOCHS
                println("      Epoch $epoch, Avg Loss: $avg_epoch_loss")
            end
        end

        println("    Evaluating fold $k...")
        y_pred_logits = model(test_x) 
        y_pred_labels = onecold(y_pred_logits, CLASS_LABELS) 

        try
            acc, err, sens, spec, prec, npv, f1, cm_fold = confusionMatrix(
                y_pred_labels,    
                test_labels_fold,
                CLASS_LABELS     
            )

            push!(fold_results, (
                acc=acc, err=err, sens=sens, spec=spec, prec=prec, npv=npv, f1=f1
            ))
            confusion_matrix_total .+= cm_fold

        catch e
             @error "Error calculating metrics for fold $k, config $(cnn_config["id"]). Skipping fold metrics. Error: $e"
             push!(fold_results, (
                 acc=NaN, err=NaN, sens=NaN, spec=NaN, prec=NaN, npv=NaN, f1=NaN
             ))
        end

        train_x = test_x = train_y = test_y = model = nothing
        GC.gc()

    end 

    if isempty(fold_results)
        @warn "No valid fold results for config $(cnn_config["id"]). Returning NaNs."
         return (
             (NaN, NaN), (NaN, NaN), (NaN, NaN), (NaN, NaN),
             (NaN, NaN), (NaN, NaN), (NaN, NaN),
             zeros(Float64, num_classes, num_classes)
         )
    end
    
    metrics_means = Dict{Symbol, Float64}()
    metrics_stds = Dict{Symbol, Float64}()
    metric_keys = keys(fold_results[1])

    for key in metric_keys
        values = [fr[key] for fr in fold_results if !isnan(fr[key])]
        if isempty(values)
            metrics_means[key] = NaN
            metrics_stds[key] = NaN
        else
             metrics_means[key] = mean(values)
             metrics_stds[key] = std(values)
        end
    end


    println("    Finished Config ID: $(cnn_config["id"]) -> Avg F1: $(round(metrics_means[:f1], digits=4)), Avg Acc: $(round(metrics_means[:acc], digits=4))")

    return (
        (metrics_means[:acc], metrics_stds[:acc]),
        (metrics_means[:err], metrics_stds[:err]),
        (metrics_means[:sens], metrics_stds[:sens]),
        (metrics_means[:spec], metrics_stds[:spec]),
        (metrics_means[:prec], metrics_stds[:prec]),
        (metrics_means[:npv], metrics_stds[:npv]),
        (metrics_means[:f1], metrics_stds[:f1]),
        confusion_matrix_total
    )
end


# --- Main Experiment Loop ---
println("\n--- STARTING CNN CROSS-VALIDATION EXPERIMENT ($K_FOLDS folds) ---")

n_samples = length(all_images)
if n_samples < K_FOLDS
    error("Number of samples ($n_samples) is less than K ($K_FOLDS). Reduce K_FOLDS.")
end

cv_indices = crossvalidation(all_labels, K_FOLDS)

results_df_cnn = DataFrame(
    ConfigID = Int[], Architecture = String[],
    Accuracy_Mean = Float64[], Accuracy_Std = Float64[],
    ErrorRate_Mean = Float64[], ErrorRate_Std = Float64[],
    Sensitivity_Mean = Float64[], Sensitivity_Std = Float64[],
    Specificity_Mean = Float64[], Specificity_Std = Float64[],
    Precision_Mean = Float64[], Precision_Std = Float64[],
    NPV_Mean = Float64[], NPV_Std = Float64[],
    F1_Mean = Float64[], F1_Std = Float64[],
    ConfusionMatrix_Sum = Array{Float64, 2}[]
)

# Itera sobre cada configuración de arquitectura CNN definida
for cnn_config in cnn_architectures
    # Ejecuta la validación cruzada para la configuración actual
    results = cnn_cross_validation(cnn_config, all_images, all_labels, cv_indices)

    ((acc_mean, acc_std), (err_mean, err_std), (sens_mean, sens_std),
     (spec_mean, spec_std), (prec_mean, prec_std), (npv_mean, npv_std),
     (f1_mean, f1_std), conf_matrix_sum) = results
    
    # Añade los resultados al DataFrame
    arch_str = join(["$k=$v" for (k, v) in cnn_config if k != "id"], ", ")
    push!(results_df_cnn, (
        cnn_config["id"], arch_str,
        acc_mean, acc_std, err_mean, err_std,
        sens_mean, sens_std, spec_mean, spec_std,
        prec_mean, prec_std, npv_mean, npv_std,
        f1_mean, f1_std,
        collect(conf_matrix_sum) 
    ))

    @printf("  -> Result for Config %d: Accuracy: %.4f ± %.4f, F1-Score: %.4f ± %.4f\n",
            cnn_config["id"], acc_mean, acc_std, f1_mean, f1_std)

    GC.gc() 
end

# --- Reporting Results ---
println("\n\n--- CNN EXPERIMENTATION SUMMARY ---")
println("Detailed Results:")
show(stdout, results_df_cnn; allrows=true, allcols=true, show_row_number=false, eltypes=false)
println("\n")

println("Best configurations by Mean F1 Score:")
valid_results_cnn_df = filter(row -> !isnan(row.F1_Mean), results_df_cnn)

if nrow(valid_results_cnn_df) > 0
    best_results_cnn = sort(valid_results_cnn_df, :F1_Mean, rev=true)

    println("------------------------------------------------------------------------------------------------")
    @printf("%-10s | %-50s | %-12s | %-12s\n", "ConfigID", "Architecture Description", "F1 Mean", "Accuracy Mean")
    println("------------------------------------------------------------------------------------------------")
    for row in eachrow(best_results_cnn)
         params_str = row.Architecture
         if length(params_str) > 50
             params_str = params_str[1:47] * "..."
         end
         f1_val = isnan(row.F1_Mean) ? "NaN" : @sprintf("%.4f", row.F1_Mean)
         acc_val = isnan(row.Accuracy_Mean) ? "NaN" : @sprintf("%.4f", row.Accuracy_Mean)
         @printf("%-10d | %-50s | %-12s | %-12s\n",
                 row.ConfigID, params_str, f1_val, acc_val)
    end
    println("------------------------------------------------------------------------------------------------")
else
    println("No valid results found to determine best CNN configurations.")
end

# --- Save Results ---
json_filename_cnn = "resultados_cnn_experimentos.json"
try
    open(json_filename_cnn, "w") do f
        JSONTables.objecttable(f, results_df_cnn)
    end
    println("\nCNN results saved to: $json_filename_cnn")
catch e
    @error "Failed to save CNN results to JSON. Error: $e"
end

println("\nCNN Experimentation completed.")