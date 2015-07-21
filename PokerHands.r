#This code will test various machine learning algorithms on data set of poker hands

#read in the poker Training data
train <- read.csv("~/PokerHands/poker-hand-training-true.data", header=FALSE)
test <- read.csv("~/PokerHands/poker-hand-testing.data", header=FALSE)

#give the columns names
names(train) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Poker_Hand")
names(test) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Poker_Hand")

#separate the class from the attributes
trainHands <- as.factor(train$Poker_Hand)
train <- train[,1:10]

testHands <- as.factor(test$Poker_Hand)
test <- test[,1:10]

#Split training set and class vector into a partial training set and validation set with an approximately %70 to %30 ratio respectively on 25000 instances
trainPart <- train[1:18000,]
trainValid <- train[-1:-18000,]
trainHandsPart <- trainHands[1:18000]
trainHandsValid = trainHands[-1:-18000]


#############################
#Naive bayes predictor      #
#############################

#library's need for this investigation
library(e1071)
library(caret)

modelNaive <- naiveBayes(trainHandsPart~., data=trainPart)
pred <- predict(modelNaive, newdata=trainValid, type="class")#run on validation as test data is heavy on cpu
NaiveBayesCM <- confusionMatrix(pred, trainHandsValid)
NaiveBayesCM 


##############################
# K- NN                   #
##############################

#library's need for this investigation
library(class)

pred <-knn(train, test, trainHands, k = 10, l = 0, prob = FALSE, use.all = TRUE)
knnCM <- confusionMatrix(pred,testHands)
knnCM

#plot the accuracy of the models
a <- c(knnCM[[3]][1],NaiveBayesCM[[3]][1])
names(a)<- c("knn","NaiveBayes")
barplot(a, main= "Accuracy of the Models in Predicting Poker Hands", xlab="Models",ylab="Accuracy % ")
