library(tidyverse)
library(tidymodels)
library(randomForest)
train <- read.csv("train_class.csv")
#convert to factor
train$winner <- as.factor(train$winner)
# remove rows w/ missing values
train <- train %>% drop_na()
features <- setdiff(names(train), c('id', 'name', 'total_votes', 'winner'))
model <- randomForest(winner ~ ., data = train[, c(features, 'winner')])
test <- read.csv("test_class.csv")
# find missing IDs
train_ids <- train$id
test_ids <- test$id
missing_ids <- setdiff(test_ids, train_ids)
test_missing <- test %>% filter(id %in% missing_ids)
test_missing <- test_missing %>% select(all_of(features))
test_missing[is.na(test_missing)] <- 0
preds <- predict(model, test_missing)
results <- data.frame(id = test %>% filter(id %in% missing_ids) %>% pull(id), winner = preds)
write.csv(results, "prediction1.csv", row.names = FALSE)
