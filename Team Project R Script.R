source("DataAnalyticsFunctions.R")
library(tidyr)
library(ggplot2)
library(tidyverse)
library(glmnet)

##Load the dataset
IBM <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

###Data Cleaning 
#Creat dummy variable for Attrition, MarritalSatus, BusinessTravel
IBM <-
  IBM %>% 
  mutate(Attrition_code = case_when(Attrition=="Yes"~1,Attrition=="No"~0)) %>%
  mutate(Married = case_when(MaritalStatus=="Married"~1,TRUE~0)) %>%
  mutate(Divorced = case_when(MaritalStatus=="Divorced"~1,TRUE~0)) %>%
  mutate(Male = case_when(Gender=="Male"~1, TRUE ~ 0)) %>%
  mutate(BusinessTravelFrequency = case_when(BusinessTravel=="Non-Travel"~0, 
                                             BusinessTravel=="Travel_Rarely"~1,
                                             BusinessTravel=="Travel_Frequently"~2))

#Remove columns that have all same value
IBM$Over18 <- NULL
IBM$StandardHours <- NULL
IBM$EmployeeCount <- NULL
IBM$EmployeeNumber <- NULL
#Remove duplicated columns
IBM$BusinessTravel <- NULL
IBM$Gender <- NULL
IBM$MaritalStatus <- NULL

###Visualization
#Graph 1.
graph.hourlyrate <- IBM %>% 
  ggplot(aes(x = Attrition, y = HourlyRate, fill = Attrition)) + geom_boxplot()
graph.hourlyrate

#Graph 2.
graph.role <- IBM %>%
  ggplot(aes(x=JobRole, fill=Attrition)) + geom_bar() + 
  theme(legend.position="right",axis.text.x=element_text(angle=60, hjust=1))
graph.role

#Graph 3.
plot(factor(Attrition) ~ OverTime, data=IBM, col=c("#FF6666","#99FFFF"), ylab="Attrition")

#Graph 4.
plot(factor(Attrition) ~ WorkLifeBalance, data=IBM, col=c("#FF6666","#99FFFF"), ylab="Attrition")

#Modify OverTime record from "Yes" and "No" to 1 and 0
IBM$OverTime <-
  case_when(IBM$OverTime == "Yes" ~1,
            IBM$OverTime =="No" ~ 0)
#Removed Attrition
IBM$Attrition <- NULL

###Classification
##Clustering
#Cluster by YearsAtCompany and YearsSinceLastPromotion
simple <- IBM[,c("YearsAtCompany","YearsSinceLastPromotion")]
par(mar = c(5, 5, 3, 1))
#Standardized the data
Ssimple <- scale(simple)
apply(Ssimple,2,sd) #sd=1
apply(Ssimple,2,mean) #mean = 0
#computing kmeans
Ssimple_kmeans <- kmeans(Ssimple,4,nstart=10)
colorcluster <- 1+Ssimple_kmeans$cluster
plot(Ssimple, col = 1, xlab="YearsAtCompany (Standardized)", ylab="YearsSinceLastPromotion (Standardized)")
plot(Ssimple, col = colorcluster,  xlab="YearsAtCompany (Standardized)", ylab="YearsSinceLastPromotion (Standardized)")
points(Ssimple_kmeans$centers, col = 1, pch = 24, cex = 1.5, lwd=1, bg = 2:5)
#displays the k centers (good to interpret)
Ssimple_kmeans$centers
#displays the size of each cluster
Ssimple_kmeans$size

#Cluster by TotalWorkingYears and MonthlyIncome
simple <- IBM[,c("TotalWorkingYears","MonthlyIncome")]
par(mar = c(5, 5, 3, 1))
#Standardized the data
Ssimple <- scale(simple)
apply(Ssimple,2,sd) #sd=1
apply(Ssimple,2,mean) #mean = 0
#computing kmeans
Ssimple_kmeans <- kmeans(Ssimple,4,nstart=10)
colorcluster <- 1+Ssimple_kmeans$cluster
plot(Ssimple, col = 1, xlab="TotalWorkingYears (Standardized)", ylab="MonthlyIncome (Standardized)")
plot(Ssimple, col = colorcluster,  xlab="TotalWorkingYears (Standardized)", ylab="MonthlyIncome (Standardized)")
points(Ssimple_kmeans$centers, col = 1, pch = 24, cex = 1.5, lwd=1, bg = 2:5)
#displays the k centers (good to interpret)
Ssimple_kmeans$centers
#displays the size of each cluster
Ssimple_kmeans$size


###Variables Selection
##Intuitive Model K-fold
nfold<-10
n <- nrow(IBM)
I.OOS <- data.frame(IOOS=rep(NA,nfold),ACC=rep(NA,nfold))
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  glm1 <- glm(Attrition_code ~ EnvironmentSatisfaction+ as.factor(OverTime) +
                WorkLifeBalance +as.factor(JobRole) +YearsAtCompany +
                DistanceFromHome + BusinessTravelFrequency,
              data=IBM,family = "binomial",subset= train)
  predglm1 <- predict(glm1, newdata=IBM[-train,],type="response")
  
  values <- FPR_TPR( (predglm1 >= 0.3) , IBM$Attrition_code[-train] )
  I.OOS$ACC[k] <- values$ACC
  I.OOS$IOOS[k] <- R2(y=IBM$Attrition_code[-train], pred=predglm1,family="binomial")
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}
Intuitive_R2 <- mean(I.OOS$IOOS)
Intuitive_R2
Intuitive_ACC <- mean(I.OOS$ACC)
Intuitive_ACC

##Features Selection by Lasso
Mx<- model.matrix(Attrition_code ~., data=IBM)[,-1]
My<- IBM$Attrition_code==1
num.features <- ncol(Mx)
num.n <- nrow(Mx)
num.churn <- sum(My)

nfold <- 10 #10 folds
n <- nrow(IBM) #the number of observations
#create a vector of fold memberships (random order)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
w <- (num.churn/num.n)*(1-(num.churn/num.n))

#lassoTheory, lasso, lassoCV
lambda.theory <- sqrt(w*log(num.features/0.05)/num.n)
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory)
lasso <- glmnet(Mx,My, family="binomial")
lassoCV <- cv.glmnet(Mx,My, family="binomial")
#features selection
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
features.1se <- support(lasso$beta[,which.min( (lassoCV$lambda-lassoCV$lambda.1se)^2)])
length(features.1se) 
features.theory <- support(lassoTheory$beta)
length(features.theory)
data.min <- data.frame(Mx[,features.min],My)
data.1se <- data.frame(Mx[,features.1se],My)
data.theory <- data.frame(Mx[,features.theory],My)
PL.OOS <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
OOS <- data.frame(logistic.interaction=rep(NA,nfold),logistic=rep(NA,nfold),null=rep(NA,nfold))

##k-fold cross validation: R-squared
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  #CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  if ( length(features.1se) == 0){  r1se <- glm(Attrition_code~1, data=IBM, subset=train, family="binomial") 
  } else {r1se <- glm(My~., data=data.1se, subset=train, family="binomial")
  }
  
  if ( length(features.theory) == 0){ 
    rtheory <- glm(Attrition_code~1, data=IBM, subset=train, family="binomial") 
  } else {rtheory <- glm(My~., data=data.theory, subset=train, family="binomial") }
  
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.theory[-train,], type="response")
  PL.OOS$PL.min[k] <- R2(y=My[-train], pred=predmin, family="binomial")
  PL.OOS$PL.1se[k] <- R2(y=My[-train], pred=pred1se, family="binomial")
  PL.OOS$PL.theory[k] <- R2(y=My[-train], pred=predtheory, family="binomial")
  
  #CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  L.OOS$L.min[k] <- R2(y=My[-train], pred=predlassomin, family="binomial")
  L.OOS$L.1se[k] <- R2(y=My[-train], pred=predlasso1se, family="binomial")
  L.OOS$L.theory[k] <- R2(y=My[-train], pred=predlassotheory, family="binomial")
  
  #Other Models
  model.logistic.interaction <-glm(Attrition_code~(.)^2, data=IBM, subset=train, family="binomial")
  model.logistic <-glm(Attrition_code~., data=IBM, subset=train,family="binomial")
  model.nulll <-glm(Attrition_code~1, data=IBM, subset=train,family="binomial")
  #get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=IBM[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=IBM[-train,], type="response")
  pred.null <- predict(model.nulll, newdata=IBM[-train,], type="response")
  
  OOS$logistic.interaction[k] <- R2(y=IBM$Attrition_code[-train], pred=pred.logistic.interaction, family="binomial")
  OOS$logistic[k] <- R2(y=IBM$Attrition_code[-train], pred=pred.logistic, family="binomial")
  OOS$null[k] <- R2(y=IBM$Attrition_code[-train], pred=pred.null, family="binomial")
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}

##R-squared
R2performance <- cbind(PL.OOS,L.OOS,OOS)
R2performance
par( mar=  c(8, 4, 4, 2) + 0.6 )
barplot(colMeans(R2performance), las=2,xpd=FALSE, ylim=c(0,.4) , xlab="", ylab = bquote( "Average Out of Sample " ~ R^2))
colMeans(R2performance)
k<- 1
val <- 0.3

##k-fold cross validation: Accuracy
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  #CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  if ( length(features.1se) == 0){  r1se <- glm(Attrition_code~1, data=IBM, subset=train, family="binomial") 
  } else {r1se <- glm(My~., data=data.1se, subset=train, family="binomial") }
  
  if ( length(features.theory) == 0){ 
    rtheory <- glm(Attrition_code~1, data=IBM, subset=train, family="binomial") 
  } else {rtheory <- glm(My~., data=data.theory, subset=train, family="binomial") }
  
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.theory[-train,], type="response")
  PL.OOS$PL.min[k] <- R2(y=My[-train], pred=predmin, family="binomial")
  PL.OOS$PL.1se[k] <- R2(y=My[-train], pred=pred1se, family="binomial")
  PL.OOS$PL.theory[k] <- R2(y=My[-train], pred=predtheory, family="binomial")
  
  values <- FPR_TPR( (predmin >= val) , My[-train] )
  PL.OOS$PL.min[k] <- values$ACC
  values <- FPR_TPR( (pred1se >= val) , My[-train] )
  PL.OOS$PL.1se[k] <- values$ACC
  values <- FPR_TPR( (predtheory >= val) , My[-train] )
  PL.OOS$PL.theory[k] <- values$ACC
  
  ### This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  
  
  values <- FPR_TPR( (predlassomin >= val) , My[-train] )
  PL.OOS$PL.min[k] <- values$ACC
  values <- FPR_TPR( ( predlasso1se >= val) , My[-train] )
  PL.OOS$PL.1se[k] <- values$ACC
  values <- FPR_TPR( (predlassotheory >= val) , My[-train] )
  PL.OOS$PL.theory[k] <- values$ACC
  
  #Other Models
  model.logistic.interaction <-glm(Attrition_code~(.)^2, data=IBM, subset=train, family="binomial")
  model.logistic <-glm(Attrition_code~., data=IBM, subset=train,family="binomial")
  model.nulll <-glm(Attrition_code~1, data=IBM, subset=train,family="binomial")
  ## get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=IBM[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=IBM[-train,], type="response")
  pred.null <- predict(model.nulll, newdata=IBM[-train,], type="response")
  #Accuracy
  values <- FPR_TPR( (pred.logistic.interaction >= val) , My[-train] )
  OOS$logistic.interaction[k] <- values$ACC
  # Logistic
  values <- FPR_TPR( (pred.logistic >= val) , My[-train] )
  OOS$logistic[k] <- values$ACC
  #Null
  values <- FPR_TPR( (pred.null >= val) , My[-train] )
  OOS$null[k] <- values$ACC
  print(paste("Iteration",k,"of",nfold,"completed"))
}

##Accuracy
ACCperformance <- cbind(PL.OOS,L.OOS, OOS)
barplot(colMeans(ACCperformance), las=2,xpd=FALSE, ylim=c(.1,1) , xlab="", ylab = bquote( "Average ACC Performance "))
colMeans(ACCperformance)
barplot(ACCperformance$PL.theory, las=2,xpd=FALSE, ylim=c(0,1) , xlab="", ylab = bquote( "Average ACC Performance "), main='K-fold Performance of Post Lasso 1se')
barplot(R2performance$PL.theory, las=2,xpd=FALSE, ylim=c(0,0.4) , xlab="", ylab = bquote( "Average Out of Sample " ~ R^2), main='K-fold Performance of Post Lasso 1se')

