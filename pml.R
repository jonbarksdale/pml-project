setwd('~/dev/pml-project/')
workDir <- "work"
if(!file.exists(workDir)) {
  dir.create(workDir)
}

downloadDataSet <- function(url, fileName) {
  dataPath <- paste(workDir, fileName, sep = '/')
  
  if(!file.exists(dataPath)) {
    download.file(url, dataPath, method = 'curl')
  }
  return(dataPath)
}

trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

fullTrainingSet <- read.csv(downloadDataSet(trainingUrl, 'pml-training.csv'))
testingSet <- read.csv(downloadDataSet(testingUrl, 'pml-testing.csv'))

library(caret)
library(rpart)

# Doing CV within the train function instead
set.seed(4518)
trainingIndex <- createDataPartition(fullTrainingSet$classe, p = 0.8, list = F)

trainSet <- fullTrainingSet[trainingIndex, ]
cvSet <- fullTrainingSet[-trainingIndex, ]

#str(trainSet[trainSet$new_window == "no", ])

#names(trainSet)
#grep("^roll_", names(trainSet), value = T)

dimensions <- c("belt", "arm", "dumbbell", "forearm")

predictors <- "user_name"
genericPredictors <- c("roll_$$", "pitch_$$", "yaw_$$", "total_accel_$$", "gyros_$$_x", "gyros_$$_y", "gyros_$$_z", "accel_$$_x", "accel_$$_y", "accel_$$_z", "magnet_$$_x", "magnet_$$_y", "magnet_$$_z")
for(k in dimensions) {
  predictors <- c(predictors, gsub("$$", k, genericPredictors, fixed=T))
}

trainSet <- trainSet[, c("classe", predictors)]

featurePlot(trainSet[, predictors[-1]], trainSet$classe, plot = 'pairs')


#trainSet <- fullTrainingSet[, c("classe", predictors)]
#summary(trainSet)

#install.packages("doParallel")
library(doParallel)
registerDoParallel(cores=4)

# Try some models! 
cachedModel <- function(fileName, fun) {
  if(!file.exists(fileName)) {
    var <- fun()
    saveRDS(var, file=fileName)
    return(var)
  } else {
    model <- readRDS(fileName)
    return(model)
  }
}

corM <- abs(cor(trainSet[, -c(1,2)]))
diag(corM) <- 0
dim(which(corM >.9, arr.ind = T)) # High correlation between 22 variables, might be worth reducing using pca 

trCtrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = T, classProbs = T, preProcOptions = list(thresh = 0.8, ICAcomp = 3))

svmLinear <- cachedModel("work/svmLinear.rds", function() {set.seed(1246); train(classe ~ ., trainSet, method = "svmLinear", trControl = trCtrl)})
svmLinearPCA <- cachedModel("work/svmLinearPCA.rds", function() {set.seed(42235); train(classe ~ ., trainSet, method = "svmLinear", preProcess = "pca", trControl = trCtrl)})
randomFModel <- cachedModel("work/randomForest.rds", function() {set.seed(2314); train(classe ~ ., trainSet, method = "rf", trControl = trCtrl)})
randomFModelPCA <- cachedModel("work/randomForestPCA.rds", function() {set.seed(98754); train(classe ~ ., trainSet, method = "rf", preProcess = "pca", trControl = trCtrl)})
adaBoost <- cachedModel("work/adaboost.rds", function() {set.seed(889); train(classe ~ ., trainSet, method = "AdaBoost.M1", trControl = trCtrl)})
adaBoostPCA <- cachedModel("work/adaboostPCA.rds", function() {set.seed(7139); train(classe ~ ., trainSet, method = "AdaBoost.M1", preProcess = "pca", trControl = trCtrl)})
rpart <- cachedModel("work/rpart.rds", function() {set.seed(3451); train(classe ~ ., trainSet, method = "rpart", trControl = trCtrl)})

evaluateModel <- function(model) {
  confusionMatrix(predict(model, cvSet), cvSet$classe)
}

models <- c('svmLinear', 'randomFModel', 'randomFModelPCA', 'adaBoost', 'adaBoostPCA', 'rpart')

results <- numeric()
for (m in models) {
  results[m] <- evaluateModel(get(m))$overall["Accuracy"]
}

bestResult <- results[which.max(results)]
bestModel <- get(names(bestResult))

print(bestResult)

if(!file.exists("answers")) {dir.create("answers")}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("answers/","problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

answers <- predict(randomFModel, testingSet)

pml_write_files(answers)
# TODO: explore the data and select a model.  Can use cv to try different models and pick the best one. 

# If only we were using the full groups....
# library(dplyr)
# groupClasses <- fullTrainingSet %>% group_by(num_window, classe) %>% tally()
# set.seed(6861)
# trainingIndex <- createDataPartition(groupClasses$classe, p=0.8, list =F )
# 
# trainingGroups <- groupClasses[trainingIndex, ]
# testGroups <- groupClasses[-trainingIndex, ]
# 
# trainingSet <- fullTrainingSet[fullTrainingSet$num_window %in% trainingGroups$num_window, ]
# cvSet <- fullTrainingSet[fullTrainingSet$num_window %in% testGroups$num_window, ]
