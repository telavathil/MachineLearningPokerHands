#an exploration of Machine learning Algorithms in the caret package
#on the Poker Hands dataset
#this attempt uses the caret pack to expore the tuning options of the models choosen
#this attempt will create a training set with equal populations for all classes to resolve the class population imbalance inherant in poker hands, but they will be tested against a sample that refects those imbalances
#this attempt will creat separte classifiers for each class and attempt to combine them into one classifier in a One-vs.-rest strategy
#These classifiers will be built with Single Vector Machines

#libraries need for this exploration
library(e1071)
library(caret)
library(NLP)
library(tm)
library(pROC)
library(lattice)
library(ada)
library(rpart)
library(kernlab)
library(ipred)
library(plyr)
library(MASS)
library(pamr)
library(randomForest)
library(ggplot2)
library(som)

###################################################################################
#initial setup for data analysis
#reading the datasets, creat and format the training and test datasets

#read in the poker data sets and store them in two datasets
train <- read.csv("~/PokerHands/poker-hand-training-true.data", header=FALSE)
test <- read.csv("~/PokerHands/poker-hand-testing.data", header=FALSE)

#give the columns names
names(train) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Hand")
names(test) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Hand")

#convert the the Class variable Hand into a factor
train$Hand <- as.factor(train$Hand)
test$Hand <- as.factor(test$Hand)

#create a training set with equal populations for all classes to resolve the class imbalance inherant in poker hands
downSampleTrain <- downSample(train[,1:10], train$Hand, list = FALSE, yname = "Hand")
upSampleTrain <- upSample(train[,1:10], train$Hand, list = FALSE, yname = "Hand")

#separate the attributes from the labels for model training
x <- upSampleTrain[,1:10]
y <- as.factor(upSampleTrain$Hand)



#create a list for storing analysis
analysis <- rep(list(Model = list(NULL),
                 Predictions = list(NULL,NULL),
                 ConfusionTables = list(NULL,NULL),
                 ConfusionMatrix = NULL
                 )
      


#give each instance a rankSum and suitSum
for(i in 1:dim(upSampleTrain)[1]){
  upSampleTrain$rankSum[i] <- sum(upSampleTrain[i,c(2,4,6,8,10)])
  upSampleTrain$suitSum[i] <- sum(upSampleTrain[i,c(1,3,5,7,9)])
}


##################################################################################
# plot the initial states of the training data

#plot  all the hands by Ranks and Suits
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|upSampleTrain$Hand,
       main="Scatterplots by Hands", 
       ylab="Sum of Rank Values", 
       xlab="Sum of Suits Values")

#display difference in training class frequencey
#par(mfrow=c(1,3))
barchart(table(train$Hand),main="Hand distribution")
barchart(table(downSampleTrain$Hand),main="Hand distribution")
barchart(table(upSampleTrain$Hand),main="Hand distribution after Up Sampling")

#function to apply a weighted transformation of a prediction vector

weightedPredict <- function(vectorPred = c(1:10)){
  value <- 0
  vectorPred[1] <- as.numeric(vectorPred[1])*(1-.50117739)
  vectorPred[2] <- as.numeric(vectorPred[2])*(1-.42256903)
  vectorPred[3] <- as.numeric(vectorPred[3])*(1-.04753902)
  vectorPred[4] <- as.numeric(vectorPred[4])*(1-.02112845)
  vectorPred[5] <- as.numeric(vectorPred[5])*(1-.00392465)
  vectorPred[6] <- as.numeric(vectorPred[6])*(1-.0019654)
  vectorPred[7] <- as.numeric(vectorPred[7])*(1-.00144058)
  vectorPred[8] <- as.numeric(vectorPred[8])*(1-.00024010)
  vectorPred[9] <- as.numeric(vectorPred[9])*(1-.00001385)
  vectorPred[10] <- as.numeric(vectorPred[10])*(1-.00000154)
  return (max(vectorPred))

}

########################################################################
#creat a Training and test set for Class 0
upSampleTrain$HandClass[upSampleTrain$Hand == 0] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 0] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 0] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 0] <- FALSE

testClass0 <- test
testClass0$HandClass[test$Hand == 0] <- TRUE
testClass0$HandClass[test$Hand != 0] <- FALSE

#plot Class 0
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class0: 'Nothing'"
       )

#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]#upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])#upSampleTrain[,12]

testpredictors<-testClass0[,1:10]
testLabels<-testClass0[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(mfinal=100, maxdepth = 5)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="treebag",
                    metric = "Kappa"
                    )

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
         response= as.numeric(as.logical(testLabels))),
     col="blue",
     main="ROC Curve for Model for Class 0"
     )


#store model, predictions, and confusion matrixes
analysis[[1]] <-list(Model = NULL,
                   Predictions = list(eval.predtrain,eval.predtest),
                   ConfusionTables = list(confTrain,confTest),
                   ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
                   )


###########################################################################
#creat a Training and test set for Class 1
upSampleTrain$HandClass[upSampleTrain$Hand == 1] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 1] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 1] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 1] <- FALSE

testClass1 <- test
testClass1$HandClass[test$Hand == 1] <- TRUE
testClass1$HandClass[test$Hand != 1] <- FALSE

#plot Class 1
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class1: 'Pair'"
)

#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]#upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])#upSampleTrain[,12]

testpredictors<-testClass1[,1:10]
testLabels<-testClass1[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(mfinal=100, maxdepth = 5)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="treebag",
                    metric = "Kappa"
                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
        response= as.numeric(as.logical(testLabels))),
    col="blue",
    main="ROC Curve for Model for Class 1"
)


#store model, predictions, and confusion matrixes
analysis[[2]] <-list(Model = NULL,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 2
upSampleTrain$HandClass[upSampleTrain$Hand == 2] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 2] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 2] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 2] <- FALSE

testClass2 <- test
testClass2$HandClass[test$Hand == 2] <- TRUE
testClass2$HandClass[test$Hand != 2] <- FALSE

#plot Class 2
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class2: 'Pair'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass2[,1:10]
testLabels<-testClass2[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(mfinal=100, maxdepth = 5)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="treebag",
                    metric = "Kappa"
                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
        response= as.numeric(as.logical(testLabels))),
    col="blue",
    main="ROC Curve for Model for Class 2"
)


#store model, predictions, and confusion matrixes
analysis[[3]] <-list(Model = NULL,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#create a Training and test set for Class 3
upSampleTrain$HandClass[upSampleTrain$Hand == 3] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 3] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 3] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 3] <- FALSE

testClass3 <- test
testClass3$HandClass[test$Hand == 3] <- TRUE
testClass3$HandClass[test$Hand != 3] <- FALSE

#plot Class 3
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class3: 'Three of a kind'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass3[,1:10]
testLabels<-testClass3[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(iter = 50, maxdepth = 5 , nu = 1)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="ada",
                    trControl=fitControl, 
                    tuneGrid=tgrid
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
        response= as.numeric(as.logical(testLabels))),
    col="blue",
    main="ROC Curve for Model for Class 3"
)


#store model, predictions, and confusion matrixes
analysis[[4]] <-list(Model = NULL,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 4
upSampleTrain$HandClass[upSampleTrain$Hand == 4] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 4] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 4] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 4] <- FALSE

testClass4 <- test
testClass4$HandClass[test$Hand == 4] <- TRUE
testClass4$HandClass[test$Hand != 4] <- FALSE

#plot Class 4
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class4: 'Straight'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass4[,1:10]
testLabels<-testClass4[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(iter = 150, maxdepth = 2 , nu = 1)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method = "ada",
                    metric = "Kappa",
                    trControl=fitControl, 
                    tuneGrid=tgrid
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
        response= as.numeric(as.logical(testLabels))),
    col="blue",
    main="ROC Curve for Model for Class 4"
)


#store model, predictions, and confusion matrixes
analysis[[5]] <-list(Model = NULL,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 5
upSampleTrain$HandClass[upSampleTrain$Hand == 5] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 5] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 5] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 5] <- FALSE

testClass5 <- test
testClass5$HandClass[test$Hand == 5] <- TRUE
testClass5$HandClass[test$Hand != 5] <- FALSE

#plot Class 5
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class5: 'Flush'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass5[,1:10]
testLabels<-testClass5[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(mtry = 3)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="treebag",
                    metric = "Kappa"
                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
        response= as.numeric(as.logical(testLabels))),
    col="blue",
    main="ROC Curve for Model for Class 5"
)


#store model, predictions, and confusion matrixes
analysis[[6]] <-list(Model = NULL,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 6
upSampleTrain$HandClass[upSampleTrain$Hand == 6] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 6] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 6] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 6] <- FALSE

testClass6 <- test
testClass6$HandClass[test$Hand == 6] <- TRUE
testClass6$HandClass[test$Hand != 6] <- FALSE

#plot Class 6
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class6: 'Full house'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass6[,1:10]
testLabels<-testClass6[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(size = 6, decay =0.001)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="nnet",
                    metric="Kappa",
                    trControl=fitControl, 
                    tuneGrid=tgrid
                                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
        response= as.numeric(as.logical(testLabels))),
    col="blue",
    main="ROC Curve for Model for Class 6"
)


#store model, predictions, and confusion matrixes
analysis[[7]] <-list(Model = eval.model,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 7
upSampleTrain$HandClass[upSampleTrain$Hand == 7] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 7] <- FALSE

downSampleTrain$
HandClass[downSampleTrain$Hand == 7] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 7] <- FALSE

testClass7 <- test
testClass7$HandClass[test$Hand == 7] <- TRUE
testClass7$HandClass[test$Hand != 7] <- FALSE

#plot Class 7
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class7: 'Four of a kind'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass7[,1:10]
testLabels<-testClass7[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(size = 3, decay =0.001)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="nnet",
                    trControl=fitControl, 
                    tuneGrid=tgrid
                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
         response= as.numeric(as.logical(testLabels))),
     col="blue",
     main="ROC Curve for Model for Class 7"
)


#store model, predictions, and confusion matrixes
analysis[[8]] <-list(Model = eval.model,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 8
upSampleTrain$HandClass[upSampleTrain$Hand == 8] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 8] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 8] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 8] <- FALSE

testClass8 <- test
testClass8$HandClass[test$Hand == 8] <- TRUE
testClass8$HandClass[test$Hand != 8] <- FALSE

#plot Class 8
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class8: 'Straight flush'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass8[,1:10]
testLabels<-testClass8[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(mtry=3)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="rf",
                    trControl=fitControl, 
                    tuneGrid=tgrid
                    
)


#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
#plot(roc(predictor=as.numeric(eval.predtest),
#         response= as.numeric(as.logical(testLabels))),
#     col="blue",
#     main="ROC Curve for Model for Class 8"
#)


#store model, predictions, and confusion matrixes
analysis[[9]] <-list(Model = eval.model,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)


###########################################################################
#creat a Training and test set for Class 9
upSampleTrain$HandClass[upSampleTrain$Hand == 9] <- TRUE
upSampleTrain$HandClass[upSampleTrain$Hand != 9] <- FALSE

downSampleTrain$HandClass[downSampleTrain$Hand == 9] <- TRUE
downSampleTrain$HandClass[downSampleTrain$Hand != 9] <- FALSE

testClass9 <- test
testClass9$HandClass[test$Hand == 9] <- TRUE
testClass9$HandClass[test$Hand != 9] <- FALSE

#plot Class 9
xyplot(upSampleTrain$rankSum~upSampleTrain$suitSum|as.factor(upSampleTrain$HandClass),
       xlab="Sum of Suits as Numeric",
       ylab="Sum of Ranks as Numeric",
       main="Pattern of Class9: 'Royal flush'"
)


#train with downSample results to test models quickly then train on upSample for model testing
trainpredictors<- upSampleTrain[,1:10]
trainLabels<- as.factor(upSampleTrain[,12])

testpredictors<-testClass9[,1:10]
testLabels<-testClass9[,12]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(mtry=3)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="rf",
                    metric="Kappa",
                    trControl=fitControl, 
                    tuneGrid=tgrid
                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#confusion Maxtrix on the True values
confusionMatrix(confTest,positive='TRUE')

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])
plot(roc(predictor=as.numeric(eval.predtest),
         response= as.numeric(as.logical(testLabels))),
     col="blue",
     main="ROC Curve for Model for Class 9"
)


#store model, predictions, and confusion matrixes
analysis[[10]] <-list(Model = eval.model,
                     Predictions = list(eval.predtrain,eval.predtest),
                     ConfusionTables = list(confTrain,confTest),
                     ConfusionMatrix = confusionMatrix(confTest,positive='TRUE')
)

#######################################################################
#Build final model from Predicted vectors

#assemble data frame of predictions from classes
predictedDF <- data.frame(as.numeric(as.logical((analysis[[1]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[2]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[3]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[4]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[5]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[6]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[7]][["Predictions"]][2][[1]]))),
                          as.numeric(as.logical((analysis[[8]][["Predictions"]][2][[1]]))),                                                                                          
                          as.numeric(as.logical((analysis[[9]][["Predictions"]][2][[1]]))),                                               
                          as.numeric(as.logical((analysis[[10]][["Predictions"]][2][[1]])))
)

#build a vector of balance accuracy values for each class model
accuracyVec <- c(0.6627,0.5373,0.505767,0.564243,0.580599,0.620161,0.527801,0.509736,0.499897,5.0e-01)

#create a table of balanced accuracy weighted predictions
predictedDF <- predictedDF*accuracyVec

#give the data frame a the class variable from testing
predictedDF <- cbind(predictedDF,test$Hand)

#name the colomns
names(predictedDF) <- c("predictClass0",
                        "predictClass1",
                        "predictClass2",
                        "predictClass3",
                        "predictClass4",
                        "predictClass5",
                        "predictClass6",
                        "predictClass7",
                        "predictClass8",
                        "predictClass9",
                        "Hand"
)



#separate the data into testing and training
set.seed(12)
inTraining <- createDataPartition(predictedDF$Hand, p = .75, list = FALSE)
training <- predictedDF[ inTraining,]
testing  <- predictedDF[-inTraining,]
table(training$Hand)
table(testing$Hand)

#train with predictors
trainpredictors <- training[,1:10]
trainLabels <- as.factor(training[,11])

#testing with predictors
testpredictors <- testing[,1:10]
testLabels <- testing[,11]

#train model
#define the resampling method
fitControl <- trainControl(method = "none")


#define parameters for tuning
tgrid <- expand.grid(sigma = 5)

set.seed(12)
eval.model <- train(x=trainpredictors,
                    y=trainLabels, 
                    method ="lssvmRadial",
                    metric="Kappa",
                    trControl=fitControl, 
                    tuneGrid=tgrid
                    
)

#build predictions
eval.predtrain<-predict(eval.model,trainpredictors)
eval.predtest<-predict(eval.model,testpredictors)

#tabluate predictions against labels
confTrain<-table(Predicted=eval.predtrain,Reference=trainLabels)
confTest<-table(Predicted=eval.predtest,Reference=testLabels)

#print and plot the results
print(confTrain[c(2,1),c(2,1)])
print(confTest[c(2,1),c(2,1)])

#confusion Maxtrix on the True values
finalCM <- confusionMatrix(confTest,positive='TRUE')

barplot(finalCM$byClass[,1:2],
        beside=T,
        main="Final Model Sensitivity and Specificity of All Class Predictions", 
        #legend(rownames(CM$byClass)),
        ylim=0:1
)

#plot a heatmap of the final confusion matrix
input <- finalCM$table
input.matrix <- data.matrix(input)
input.matrix.normalized <- normalize(input.matrix)

colnames(input.matrix.normalized) <- rownames(input.matrix.normalized)


#confusion <- as.data.frame(as.table(input.matrix.normalized))
confusion <- as.data.frame(as.table(input.matrix))

plot <- ggplot(confusion)
plot + geom_tile(aes(x=Prediction, y=Reference, fill=Freq)) + scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + labs(fill="Frequency")
