#this will investigate the naive bayes classifier to optimize it predictor

#library's need for this investigation
library(e1071)

model1 <- naiveBayes(trainHandsPart~., data=trainPart, laplace= 0)
model2 <- naiveBayes(trainHandsPart~., data=trainPart, laplace= 1)
model3 <- naiveBayes(trainHandsPart~., data=trainPart, laplace= 2)
model4 <- naiveBayes(trainHandsPart~., data=trainPart, laplace= 3)
model5 <- naiveBayes(trainHandsPart~., data=trainPart, laplace= 4)

pred1 <- predict(model1, newdata=trainValid, type="class")#run on validation as test data is heavy on cpu
pred2 <- predict(model2, newdata=trainValid, type="class")
pred3 <- predict(model3, newdata=trainValid, type="class")
pred4 <- predict(model4, newdata=trainValid, type="class")
pred5 <- predict(model5, newdata=trainValid, type="class")

confusionMatrix(pred1, trainHandsValid)
confusionMatrix(pred2, trainHandsValid)
confusionMatrix(pred3, trainHandsValid)
confusionMatrix(pred4, trainHandsValid)
confusionMatrix(pred5, trainHandsValid)

NaiveBayesCM <- confusionMatrix(pred, trainHandsValid)
NaiveBayesCM 
