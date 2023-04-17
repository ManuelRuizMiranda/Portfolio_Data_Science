library(tidyverse)
library(caret)
library(utils)
library(skimr)
      
(scipen=999)

temp = tempfile()
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',temp)

temp1 = tempfile() 
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',temp1)

df_fit <- read.csv(temp)
test_fit <- read.csv(temp1)

str(df_fit)
skim(df_fit)


colMeans(is.na(df_fit))
which(colMeans(is.na(df_fit))>=0.2)
      
df_fit <- df_fit[,-which(colMeans(is.na(df_fit))>=0.2)]      

df_fit <- df_fit %>% 
  select(-starts_with(c('kurt','ske','max','min','ampli')))

df_fit <- df_fit %>% mutate(cvtd_timestamp = as.Date(cvtd_timestamp,"%d/%m/%y"))
class(df_fit$cvtd_timestamp)

df_fit <- df_fit %>% mutate(classe = as.factor(classe))
class(df_fit$classe)

skim(df_fit)

skim(test_fit)

#Encontrar variables con varianza cero

num_cols <- sapply(df_fit, is.numeric)

varianza <- nearZeroVar(df_fit[num_cols],saveMetrics = T)

table(varianza$nzv)

#Buscar variables correlacionadas

train_fit_cor <- cor(df_fit[num_cols])
eliminate <- findCorrelation(train_fit_cor,verbose = T,names = T) 
eliminate 
df_fit <- df_fit %>% select(-(eliminate))

findLinearCombos(train_fit_cor)

#Pre-procesar variables

pre_pca <- preProcess(df_fit,method = "pca",thresh = 0.8)
df_preProc <- predict(pre_pca,df_fit)
dim(df_preProc)


#Crear partición

intrain <- createDataPartition(y = df_preProc$classe,p = 0.85,list = F)

training <- df_preProc[intrain,] 
testing  <- df_preProc[-intrain,]

library(doParallel)

cl=makePSOCKcluster(5)
registerDoParallel(cl)

#Modelización

library(randomForest)

set.seed(1235)

cross_valid <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 10)

model_rf <- train(classe~.,data = training,method = "rf",
                  trControl = cross_valid)


pred <- predict(model_rf,testing)

conf_matr <- confusionMatrix(pred,testing$classe)

#Importancia de Variables

var <- varImp(model_rf,scale = FALSE)

ggplot(var, aes(x=reorder(rownames(Var)))) +
  xlab('Variable')+
  ylab('Overall Importance')+
  theme_light() +
  coord_flip() 

plot(model_rf)

#Finalmente tomamos una muestra aleatoria de 20 valores del vector de predicciones

predicciones <- head(pred,20)
 
