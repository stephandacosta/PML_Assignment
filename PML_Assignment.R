# The objective of the exercise is to predict the activity class of physical exercises conducted by 6 participants.  
# 
# We will run the analysis on the following data  
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
# 
# We first load the data. 

library(caret)
data <- read.csv("pml-training.csv")
# 
# Because activity class would be predicted only be the measurement, we decide to remove timestamp and username from the feature set.  
# Also due to unavailability of data in some columns (avg, stddev, min, max, skewness and kurtosis) we also decide to remove from the feature set.  
headers <- names(data)
non_predictors <- grep("X|user_name|timestamp|window|avg|stddev|var|min|max|skewness|kurtosis|amplitude", headers)

#We can then create a training set and a testing set
intrain <- createDataPartition(y=data$classe, p=0.7, list=FALSE)
training <- data[intrain,-non_predictors]
testing <- data[-intrain,-non_predictors]

# We chose to run Random Forest for this exercise as the algorithm would provide best accuracy for this classification problem  
modFit <- train(classe ~ ., data=training, method="rf", preProcess="pca")

# We run the prediction model to the in-sample testing data and run cross-validation table to check the error
prediction <- predict(modFit, testing)
table(testing$classe, prediction)

# prediction
# A    B    C    D    E
# A 1671    1    0    1    1
# B   11 1117   11    0    0
# C    0   16 1004    6    0
# D    0    0    8  953    3
# E    0    1    2    4 1075

sum(testing$classe==prediction)/length(testing$classe)
#[1] 0.988955

# Thus we estimate less then 2% out-of-sample error.  

# Finally we can run the prediction model to the 20 cases provided    
testing2 <- read.csv("pml-testing.csv")
prediction2 <- predict(modFit, testing2)
answers <- cbind(testing2$problem_id, as.data.frame(prediction2))

# testing2$problem_id prediction2
# 1                    1           B
# 2                    2           A
# 3                    3           A
# 4                    4           A
# 5                    5           A
# 6                    6           E
# 7                    7           D
# 8                    8           B
# 9                    9           A
# 10                  10           A
# 11                  11           B
# 12                  12           C
# 13                  13           B
# 14                  14           A
# 15                  15           E
# 16                  16           E
# 17                  17           A
# 18                  18           B
# 19                  19           B
# 20                  20           B



# create files

for(i in 1:20){
        filename = paste0("problem_id_",i,".txt")
        write.table(answers[i,"prediction2"],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}








