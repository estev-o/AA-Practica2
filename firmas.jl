
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses
using StatsBase 

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    numFeatures = length(feature)

    ordered_classes = sort(classes) 

    if numClasses == 1
        return reshape(fill(true, numFeatures), :, 1) 
    elseif numClasses == 2
        oneHot = BitArray{2}(undef, numFeatures, 2)
        oneHot[:, 1] .= (feature .== ordered_classes[1])
        oneHot[:, 2] .= (feature .== ordered_classes[2])
        return oneHot
    else 
        oneHot = BitArray{2}(undef, numFeatures, numClasses)
        for (idxClass, currentClass) in enumerate(ordered_classes)
            oneHot[:, idxClass] .= (feature .== currentClass)
        end
        return oneHot
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return oneHotEncoding(feature, [false, true])
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minValues = minimum(dataset, dims=1)
    maxValues = maximum(dataset, dims=1)
    return (minValues, maxValues)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    meanValues = mean(dataset, dims=1)
    stdValues = std(dataset, dims=1)
    return (meanValues, stdValues)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
   normalizationParameters=calculateMinMaxNormalizationParameters(dataset);
   return normalizeMinMax!(dataset,normalizationParameters);

end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    copiedDataset = copy(dataset)
    normalizeMinMax!(copiedDataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    copiedDataset = copy(dataset)
    normalizeMinMax!(copiedDataset)
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    meanValues, stdValues = normalizationParameters
    dataset .-= meanValues
    dataset ./= stdValues
    dataset[:, vec(stdValues .== 0)] .= 0
    return dataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    return normalizeZeroMean!(copy(dataset), normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizationParameters)
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end;


function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        return reshape(classifyOutputs(outputs[:]; threshold=threshold), :, 1)
    else
        return outputs .== maximum(outputs, dims=2)
    end
end;


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1 && size(targets, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        return mean(correctClassifications)
    end
end


function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .> threshold
    return accuracy(outputs, targets)
end


function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    n = size(targets,2)
    if n == 1
        return accuracy(vec(outputs), vec(targets), threshold=threshold)
    else
        outputs = classifyOutputs(outputs)
        return accuracy(outputs, targets)
    end
end



function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

   ann = Chain();
   numInputsLayer=numInputs;
   if length(topology)!=0
    i = 1;
    for numOutputsLayer = topology    
       ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]) );
       numInputsLayer = numOutputsLayer;
       i+=1;
       end
   end

   if numOutputs == 1
       ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
   else
       ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity), softmax)
   end

   return ann

end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs = dataset[1]
    inputsT = inputs'
    targets = dataset[2]
    targetsT = targets'

    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)

    loss(model, x, y) = (numOutputs == 1) ? Flux.Losses.binarycrossentropy(model(x), y) : Flux.Losses.crossentropy(model(x), y)
    ann = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)
    opt_state = Flux.setup(Adam(learningRate), ann)

    lossActual = loss(ann, inputsT, targetsT)
    losses = Float32[lossActual]

    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt_state)
        lossActual = loss(ann, inputsT, targetsT)
        push!(losses, lossActual)
        if lossActual < minLoss
            break
        end
    end

    return (ann, losses)
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    reconvertida = reshape(targets, :, 1)
    return trainClassANN(topology, (inputs, reconvertida), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate) #todo aquí también?
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    index = randperm(N);
    numTrainingInstances = Int(floor(N*(1-P)));
    return (index[1:numTrainingInstances], index[numTrainingInstances+1:end]);
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
     @assert 0.0 ≤ Pval ≤ 1.0 "Pval debe estar entre 0 y 1."
    @assert 0.0 ≤ Ptest ≤ 1.0 "Ptest debe estar entre 0 y 1."        
    @assert (Pval + Ptest) ≤ 1.0 "La suma de Pval y Ptest no puede exceder 1."
    perm = randperm(N)        
    # Calculamos el número de instancias para test
    ntest = Int(round(Ptest * N))        
    # El resto de instancias se destinará a entrenamiento + validación
    ntrain_val = N - ntest        
    # Del total original queremos que el conjunto de validación tenga Pval*N instancias
    nval = Int(round(Pval * N))        
    # El conjunto de entrenamiento será el resto (ntrain_val - nval)
    ntrain = ntrain_val - nval    
    # Se extraen los subconjuntos:
    trainingIndices   = perm[1:ntrain]
    validationIndices = perm[ntrain+1 : ntrain_val]
    testIndices       = perm[ntrain_val+1 : end]        
    return (trainingIndices, validationIndices, testIndices)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
        (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), 
        falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
        (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), 
        falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    # Desempaquetamos los datasets
    trainingInputs, trainingTargets   = trainingDataset
    validationInputs, validationTargets = validationDataset
    testInputs, testTargets             = testDataset

    # Comprobación de dimensiones
    @assert size(trainingInputs,1) == size(trainingTargets,1) "Discrepancia en número de patrones en entrenamiento."
    if size(validationInputs,1) > 0
        @assert size(validationInputs,1) == size(validationTargets,1) "Discrepancia en número de patrones en validación."
    end
    if size(testInputs,1) > 0
        @assert size(testInputs,1) == size(testTargets,1) "Discrepancia en número de patrones en test."
    end
    nfeatures = size(trainingInputs,2)
    if size(validationInputs,1) > 0
        @assert size(validationInputs,2) == nfeatures "El número de características no coincide entre entrenamiento y validación."
    end
    if size(testInputs,1) > 0
        @assert size(testInputs,2) == nfeatures "El número de características no coincide entre entrenamiento y test."
    end

    # Se crea la red; el número de entradas es el número de columnas (dado que los patrones están en filas)
    ann = buildClassANN(nfeatures, topology, size(trainingTargets,2); transferFunctions=transferFunctions)

    # Función de loss: se transpone para que cada patrón esté en una columna, tal como requiere Flux
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y)
    # Vectores para almacenar las métricas
    trainingLosses     = Float32[]
    validationLosses   = Float32[]
    testLosses         = Float32[]
    trainingAccuracies = Float32[]
    validationAccuracies = Float32[]
    testAccuracies     = Float32[]

    # Función local para calcular las métricas en cada ciclo
    function calculateMetrics()
        # Para entrenamiento
        train_loss    = loss(ann, trainingInputs', trainingTargets')
        train_outputs = ann(trainingInputs')
        train_acc     = accuracy(train_outputs', trainingTargets)

        # Para validación
        val_loss = val_acc = Inf
        if size(validationInputs, 1) > 0
            val_loss = loss(ann, validationInputs', validationTargets')
            val_outputs = ann(validationInputs')
            val_acc = accuracy(val_outputs', validationTargets)
        end

        # Para test
        tst_loss = tst_acc = Inf
        if size(testInputs, 1) > 0
            tst_loss = loss(ann, testInputs', testTargets')
            tst_outputs = ann(testInputs')
            tst_acc = accuracy(tst_outputs', testTargets)
        end

        return train_loss, train_acc, val_loss, val_acc, tst_loss, tst_acc
    end

    # Cálculo inicial (it 0)
    train_loss, train_acc, val_loss, val_acc, tst_loss, tst_acc = calculateMetrics()
    push!(trainingLosses, train_loss)
    push!(trainingAccuracies, train_acc)
    if size(validationInputs, 1) > 0
        push!(validationLosses, val_loss)
        push!(validationAccuracies, val_acc)
    end
    if size(testInputs, 1) > 0
        push!(testLosses, tst_loss)
        push!(testAccuracies, tst_acc)
    end

    numEpoch = 0
    numEpochsValidation = 0
    bestValidationLoss = val_loss
    bestANN = deepcopy(ann)

    opt_state = Flux.setup(Adam(learningRate), ann)
    # Bucle de entrenamiento con parada temprana (early stopping)
    while numEpoch < maxEpochs && train_loss > minLoss && ((size(validationInputs,1) == 0) || (numEpochsValidation < maxEpochsVal))
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state);
        numEpoch += 1
        train_loss, train_acc, val_loss, val_acc, tst_loss, tst_acc = calculateMetrics()
        push!(trainingLosses, train_loss)
        push!(trainingAccuracies, train_acc)
        if size(validationInputs, 1) > 0
            push!(validationLosses, val_loss)
            push!(validationAccuracies, val_acc)
        end
        if size(testInputs, 1) > 0
            push!(testLosses, tst_loss)
            push!(testAccuracies, tst_acc)
        end

        if size(validationInputs,1) > 0
            if val_loss < bestValidationLoss
                bestValidationLoss = val_loss
                numEpochsValidation = 0
                bestANN = deepcopy(ann)
            else
                numEpochsValidation += 1
            end
        end
    end

    if size(validationInputs, 1) == 0
        bestANN = ann
    end
    
    return (bestANN, trainingLosses, validationLosses, testLosses)
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
        (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
        (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    newDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    return trainClassANN(topology, newDataset;
        validationDataset=(validationDataset[1], reshape(validationDataset[2], :, 1)),
        testDataset=(testDataset[1], reshape(testDataset[2], :, 1)),
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal)
end




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VN = sum((outputs .== false) .& (targets .== false))
    VP = sum((outputs .== true) .& (targets .== true))
    FN = sum((outputs .== false) .& (targets .== true))
    FP = sum((outputs .== true) .& (targets .== false))

    accuracy = (VN + VP) / (VN + VP + FN + FP)

    errorRate = (FN + FP) / (VN + VP + FN + FP)

    if(VP==0&&FN==0)
        recall=1.0
    else
        recall= VP /(FN+VP)
    end

    if(VN==0&&FP==0)
        specificity=1.0
    else
        specificity= VN /(FP+VN)
    end
    
    if(VP==0&&FP==0)
        precision=1.0
    else
        precision= VP/(VP+FP)
    end

    if(VN==0&&FN==0)
        NPV=1.0
    else
        NPV= VN/(VN+FN)
    end
    
    if(recall==0&&precision==0)
        F1=0.0
    else
        F1= 2*((recall*precision)/(recall+precision))
    end

    confMatrix = [VN FP; FN VP]
    return (accuracy, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    booleanOutputs=outputs.>threshold;
    return confusionMatrix(booleanOutputs, targets)
end;

using LinearAlgebra

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    num_classes = size(outputs, 2)
    recall = zeros(num_classes)
    specificity = zeros(num_classes)
    precision = zeros(num_classes)
    NPV = zeros(num_classes)
    F1 = zeros(num_classes)
    
    confMatrix = zeros(Int, num_classes, num_classes)
    
    if(num_classes==1)
        return confusionMatrix(vec(outputs),vec(targets))
    end 

    for i in 1:num_classes
        _,_,recall[i],specificity[i],precision[i],NPV[i],F1[i] = confusionMatrix(outputs[:,i],targets[:,i])
        for j in 1:num_classes
            confMatrix[i, j] = sum((outputs[:, j]) .& (targets[:, i]))
        end
    end

    if weighted
        class_counts = Float64.(vec(sum(targets, dims=1)))
        total_samples = sum(class_counts)
        recall = sum(recall .* class_counts) / total_samples
        specificity = sum(specificity .* class_counts) / total_samples
        precision = sum(precision .* class_counts) / total_samples
        NPV = sum(NPV .* class_counts) / total_samples
        F1 = sum(F1 .* class_counts) / total_samples
    else
        recall = mean(recall)
        specificity = mean(specificity)
        precision = mean(precision)
        NPV = mean(NPV)
        F1 = mean(F1)
    end
    
    accuracy = sum(diag(confMatrix)) / sum(confMatrix)
    errorRate = 1 - accuracy
    
    return accuracy, errorRate, recall, specificity, precision, NPV, F1, confMatrix
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    outputs_clasificados =classifyOutputs(outputs;threshold=threshold)
    return confusionMatrix(outputs_clasificados, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert all([in(label, classes) for label in vcat(targets, outputs)]) "Hay etiquetas fuera de las clases permitidas"

    # Codificación one-hot de outputs y targets
    targets_encoded = oneHotEncoding(targets, classes)
    outputs_encoded = oneHotEncoding(outputs, classes)

    # Llamada a la función de confusionMatrix que trabaja con matrices booleanas
    return confusionMatrix(outputs_encoded, targets_encoded; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
   classes = unique(vcat(targets, outputs))

   outputs_onehot = oneHotEncoding(outputs, classes)
   targets_onehot = oneHotEncoding(targets, classes)

    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end;

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), reshape(targets, :, 1));
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);

printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true) = printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    printConfusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset
    trainingInputs = Float64.(trainingInputs)
    testInputs = Float64.(testInputs)

    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes=maximumNodes)
    prediction_scores = evaluateTree(model, testInputs)
    predicted_targets_bool = (prediction_scores .> 0.0)
    return predicted_targets_bool
end

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets_onehot = trainingDataset
    numClasses = size(trainingTargets_onehot, 2)
    numTestSamples = size(testInputs, 1)

    if (numClasses<=1)
        trainingTargets_bool = vec(trainingTargets_onehot)
        return trainClassDoME((trainingInputs, trainingTargets_bool), testInputs, maximumNodes)
    else
        matrix_outputs = zeros(Float64, numTestSamples, numClasses)
        for i=1:numClasses
            classTargets_bool = trainingTargets_onehot[:, i] 
            classResult_bool = trainClassDoME((trainingInputs, classTargets_bool), testInputs, maximumNodes)
            matrix_outputs[:, i] = Float64.(classResult_bool)
        end
        return matrix_outputs
    end
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets_any = trainingDataset # trainingTargets_any es Vector{String}
    numTestSamples = size(testInputs,1)
    classes = sort(unique(trainingTargets_any))
    positiveClass = classes[2] 
    negativeClass = classes[1] 

    trainingTargets_bool :: BitVector = (trainingTargets_any .== positiveClass)

    predicted_targets_bool :: BitVector = trainClassDoME((trainingInputs, trainingTargets_bool), testInputs, maximumNodes)

    testOutputs_string = Vector{String}(undef, numTestSamples)
    for i in 1:numTestSamples
        testOutputs_string[i] = predicted_targets_bool[i] ? positiveClass : negativeClass
    end

    return testOutputs_string 
end




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = 1:k

    indices_repetidos = repeat(indices, 1, ceil(Int, N / k))
    indices_tomados = indices_repetidos[1:N]
    shuffle!(indices_tomados)

    return indices_tomados
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    # Vector para almacenar los índices de fold
    indices = Array{Int64,1}(undef, length(targets))

    # Asignar los índices de cross-validation para las instancias positivas
    indices[targets .== true] = crossvalidation(sum(targets .== true), k)

    # Asignar los índices de cross-validation para las instancias negativas
    indices[targets .== false] = crossvalidation(sum(targets .== false), k)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N = size(targets, 1)
    indices = Array{Int64}(undef, N)
    
    for i in 1:size(targets, 2)
        class_indices = findall(targets[:, i])
        indices[class_indices] = crossvalidation(length(class_indices), k)
    end
    
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targets_nuevos=oneHotEncoding(targets)
    
    return crossvalidation(targets_nuevos,k)
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    inputs, targets_raw = dataset

    # Obtener las clases y codificar las salidas deseadas en one-hot
    classes = unique(targets_raw)
    targets = oneHotEncoding(targets_raw, classes)

    numClasses = size(targets, 2)
    numFolds = maximum(crossValidationIndices)

    # Inicializar vectores de métricas para cada fold
    accuracyVec = zeros(Float64, numFolds)
    errorRateVec = zeros(Float64, numFolds)
    recallVec = zeros(Float64, numFolds)
    specificityVec = zeros(Float64, numFolds)
    precisionVec = zeros(Float64, numFolds)
    vpnVec = zeros(Float64, numFolds)
    f1Vec = zeros(Float64, numFolds)

    # Inicializar matriz de confusión global
    confusionMatrixGlobal = zeros(Float64, numClasses, numClasses)

    for fold in 1:numFolds
        # Separar conjunto de entrenamiento y test según fold
        idx_train = findall(crossValidationIndices .!= fold)
        idx_test = findall(crossValidationIndices .== fold)

        inputs_train, targets_train = inputs[idx_train, :], targets[idx_train, :]
        inputs_test, targets_test = inputs[idx_test, :], targets[idx_test, :]

        # Inicializar arrays para guardar las métricas y matrices de confusión por ejecución
        acc_exec = zeros(Float64, numExecutions)
        err_exec = zeros(Float64, numExecutions)
        recall_exec = zeros(Float64, numExecutions)
        spec_exec = zeros(Float64, numExecutions)
        precision_exec = zeros(Float64, numExecutions)
        vpn_exec = zeros(Float64, numExecutions)
        f1_exec = zeros(Float64, numExecutions)
        conf_matrices_exec = zeros(Float64, numClasses, numClasses, numExecutions)

        # Ejecutar entrenamientos múltiples
        for exec in 1:numExecutions

            # Separar conjunto de validación dentro de cada ejecución
            if validationRatio > 0
                validationRatio_calculado=(validationRatio*numFolds)/(numFolds-1)
                idx_train_split, idx_val = holdOut(size(inputs_train, 1), validationRatio_calculado)
                exec_inputs_train, exec_targets_train = inputs_train[idx_train_split, :], targets_train[idx_train_split, :]
                inputs_val, targets_val = inputs_train[idx_val, :], targets_train[idx_val, :]
            else
                exec_inputs_train, exec_targets_train = inputs_train, targets_train
                inputs_val = Array{eltype(inputs_train)}(undef, 0, size(inputs_train, 2))
                targets_val = Array{Bool}(undef, 0, size(targets_train, 2))
            end
        
            # Entrenar red neuronal
            network, _, _, _ = trainClassANN(
                topology, (exec_inputs_train, exec_targets_train);
                validationDataset=(inputs_val, targets_val),
                testDataset=(inputs_test, targets_test),
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs,
                minLoss=minLoss,
                learningRate=learningRate,
                maxEpochsVal=maxEpochsVal
            )
        
            # Evaluar red neuronal
            outputs_test = network(inputs_test')
            acc, err, sens, spec, vpp, vpn, f1, conf_matrix = confusionMatrix(outputs_test', targets_test)
        
            # Guardar métricas y matrices de confusión
            acc_exec[exec] = acc
            err_exec[exec] = err
            recall_exec[exec] = sens
            spec_exec[exec] = spec
            precision_exec[exec] = vpp
            vpn_exec[exec] = vpn
            f1_exec[exec] = f1
            @assert size(conf_matrices_exec) == (numClasses, numClasses, numExecutions) "ANNCV Error: Tamaño inesperado de conf_matrices_exec ($(size(conf_matrices_exec))), se esperaba ($numClasses, $numClasses, $numExecutions)"
            # Verificar que conf_matrix sea realmente una Matriz 2D
            @assert typeof(conf_matrix) <: AbstractMatrix "ANNCV Error: conf_matrix no es una Matriz (tipo: $(typeof(conf_matrix)))"
            @assert size(conf_matrix) == (numClasses, numClasses) "ANNCV Error: Tamaño inesperado de conf_matrix ($(size(conf_matrix))), se esperaba ($numClasses, $numClasses)"
            @assert exec >= 1 && exec <= numExecutions "ANNCV Error: Índice de ejecución inválido (exec = $exec)"
            conf_matrices_exec[:, :, exec] = conf_matrix
        end
        

        # Promediar métricas y matrices de confusión por fold
        accuracyVec[fold] = mean(acc_exec)
        errorRateVec[fold] = mean(err_exec)
        recallVec[fold] = mean(recall_exec)
        specificityVec[fold] = mean(spec_exec)
        precisionVec[fold] = mean(precision_exec)
        vpnVec[fold] = mean(vpn_exec)
        f1Vec[fold] = mean(f1_exec)
        confusionMatrixGlobal .+= mean(conf_matrices_exec, dims=3)[:, :, 1]  # Promedio y suma a global
    end

    # Devolver métricas con media y desviación típica, y la matriz de confusión
    return (
        (mean(accuracyVec), std(accuracyVec)),
        (mean(errorRateVec), std(errorRateVec)),
        (mean(recallVec), std(recallVec)),
        (mean(specificityVec), std(specificityVec)),
        (mean(precisionVec), std(precisionVec)),
        (mean(vpnVec), std(vpnVec)),
        (mean(f1Vec), std(f1Vec)),
        confusionMatrixGlobal
    )
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, 
                              dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, 
                              crossValidationIndices::Array{Int64,1})

    inputs, targets_raw = dataset
    numFolds = maximum(crossValidationIndices)

    # Si es red neuronal, derivamos a ANNCrossValidation directamente
    if modelType == :ANN
        topology = modelHyperparameters["topology"]
        learningRate = get(modelHyperparameters, "learningRate", 0.01)
        validationRatio = get(modelHyperparameters, "validationRatio", 0.0)
        numExecutions = get(modelHyperparameters, "numExecutions", 50)
        maxEpochs = get(modelHyperparameters, "maxEpochs", 1000)
        maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20)
        inputs_f32 = Float32.(inputs)
        dataset_f32 = (inputs_f32, targets_raw)
        return ANNCrossValidation(topology, dataset_f32, crossValidationIndices;
                                  numExecutions=numExecutions,
                                  learningRate=learningRate,
                                  validationRatio=validationRatio,
                                  maxEpochs=maxEpochs,
                                  maxEpochsVal=maxEpochsVal)
    end

    # Preparar todo para MLJ o DoME
    targets = string.(targets_raw)   # Convertimos las etiquetas a String
    classes = unique(targets)
    numClasses = length(classes)

    # Inicializar métricas
    accVec = zeros(Float64, numFolds)
    errVec = zeros(Float64, numFolds)
    sensVec = zeros(Float64, numFolds)
    specVec = zeros(Float64, numFolds)
    precisionVec = zeros(Float64, numFolds)
    vpnVec = zeros(Float64, numFolds)
    f1Vec = zeros(Float64, numFolds)
    confusionGlobal = zeros(Float64, numClasses, numClasses)

    for fold in 1:numFolds
        testIdx = findall(crossValidationIndices .== fold)
        trainIdx = findall(crossValidationIndices .!= fold)

        X_train, y_train = inputs[trainIdx, :], targets[trainIdx]
        X_test, y_test = inputs[testIdx, :], targets[testIdx]

        # Predicciones dependiendo del modelo
        if modelType == :DoME
            maxNodes = modelHyperparameters["maximumNodes"]
            y_pred = trainClassDoME((X_train, y_train), X_test, maxNodes)   
        else
            # Crear modelo de MLJ
            if modelType == :SVC
                kernelDict = Dict("linear" => LIBSVM.Kernel.Linear,
                                  "rbf" => LIBSVM.Kernel.RadialBasis,
                                  "sigmoid" => LIBSVM.Kernel.Sigmoid,
                                  "poly" => LIBSVM.Kernel.Polynomial)
                kernelType = kernelDict[modelHyperparameters["kernel"]]
                model = SVMClassifier(
                    kernel = kernelType,
                    cost = Float64(modelHyperparameters["C"]),
                    gamma = Float64(get(modelHyperparameters, "gamma", 0.0)),
                    degree = Int32(get(modelHyperparameters, "degree", 3)),
                    coef0 = Float64(get(modelHyperparameters, "coef0", 0.0))
                )
            elseif modelType == :DecisionTreeClassifier
                model = DTClassifier(max_depth=modelHyperparameters["max_depth"],
                                     rng=Random.MersenneTwister(1))
            elseif modelType == :KNeighborsClassifier
                model = kNNClassifier(K = modelHyperparameters["n_neighbors"])
            else
                error("Modelo no soportado")
            end

            # Crear máquina y entrenar
            mach = machine(model, MLJ.table(X_train), categorical(y_train))
            MLJ.fit!(mach, verbosity=0)

            # Predecir
            y_pred = MLJ.predict(mach, MLJ.table(X_test))
            if modelType == :SVC
                y_pred = y_pred
            else
                y_pred = mode.(y_pred)  # Para árboles y kNN
            end
        end

        # Calcular métricas y matriz de confusión
        acc, err, sens, spec, vpp, vpn, f1, cm = confusionMatrix(y_pred, y_test, classes)
        accVec[fold] = acc
        errVec[fold] = err
        sensVec[fold] = sens
        specVec[fold] = spec
        precisionVec[fold] = vpp
        vpnVec[fold] = vpn
        f1Vec[fold] = f1
        confusionGlobal .+= cm
    end

    # Retornar tupla con media y std de cada métrica y la matriz de confusión global
    return (
        (mean(accVec), std(accVec)),
        (mean(errVec), std(errVec)),
        (mean(sensVec), std(sensVec)),
        (mean(specVec), std(specVec)),
        (mean(precisionVec), std(precisionVec)),
        (mean(vpnVec), std(vpnVec)),
        (mean(f1Vec), std(f1Vec)),
        confusionGlobal
    )
end








