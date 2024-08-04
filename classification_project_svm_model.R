# Load necessary libraries
library(e1071)
library(readr)
library(dplyr)
library(caret)  # For grid search and cross-validation
library(ggplot2)
library(GGally)
library(tidyr)
library(ggcorrplot)
library(pROC)  # For ROC analysis

# Load the data
train_data <- read_csv("train_class.csv")
test_data <- read_csv("test_class.csv")

# Identify the predictor columns
predictor_columns <- setdiff(names(train_data), c("winner", "id"))

# EDA Section

# Summarize the data
summary(train_data)

# Check for missing values
missing_values <- train_data %>%
  summarise_all(~ sum(is.na(.)))
print(missing_values)

# Visualize the distribution of the target variable
ggplot(train_data, aes(x = winner)) +
  geom_bar() +
  ggtitle("Distribution of Target Variable (Winner)")

# Data Preprocessing
train_data$winner <- as.factor(train_data$winner)

# Impute missing values with column means for training data
for (col in predictor_columns) {
  train_data[[col]][is.na(train_data[[col]])] <- mean(train_data[[col]], na.rm = TRUE)
}

# Impute missing values with column means for test data
for (col in predictor_columns) {
  test_data[[col]][is.na(test_data[[col]])] <- mean(test_data[[col]], na.rm = TRUE)
}

# Normalize the data
preproc <- preProcess(train_data[, predictor_columns], method = c("center", "scale"))
train_data[, predictor_columns] <- predict(preproc, train_data[, predictor_columns])
test_data[, predictor_columns] <- predict(preproc, test_data[, predictor_columns])

# Define the control with ROC
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Define the grid of hyperparameters (using sigma instead of gamma)
svm_grid <- expand.grid(sigma = 2^(-5:2), C = 2^(-5:2))

# Train the SVM model with grid search
svm_tuned <- train(
  winner ~ ., data = train_data[, c(predictor_columns, "winner")],
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  metric = "ROC"
)

# Best model
print(svm_tuned$bestTune)

# Plot the ROC curve for the best model
svm_probs <- predict(svm_tuned, train_data[, predictor_columns], type = "prob")
roc_curve <- roc(train_data$winner, svm_probs[,2], levels = rev(levels(train_data$winner)))
plot(roc_curve, main = "ROC Curve for SVM Model")

# Make predictions on the test set
predictions <- predict(svm_tuned, test_data[, predictor_columns])

# Prepare the submission file
submission <- data.frame(id = test_data$id, winner = predictions)
write_csv(submission, "submission_svm_tuned.csv")

auc_value <- auc(roc_curve)
print(paste("AUC: ", auc_value))