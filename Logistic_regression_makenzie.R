library(tidyverse)
library(tidymodels)
library(ISLR)

classification <- read_csv("train_class.csv")
col_explain <- read_csv("col_descriptions.csv")

#looking at the data set
summary(classification)
#NA's present nearing the end of the data set betw. cols pertaining to income and GDP
class_clean <- classification %>%
  filter(complete.cases(.))
# removing any duplicated columns
class_clean <- class_clean %>%
  select(-c(name, x0021e, x0024e, x0033e, x0034e, x0035e)) %>%
  mutate(winner = as.factor(winner))

set.seed(16)

# data resampling
class_split <- initial_split(class_clean, prop = 0.75, strata = winner)
class_train <- training(class_split)
class_test <- testing(class_split)

# a little EDA
table(class_train$winner)
# Biden won 279 counties, Trump 1440
# baseline predicts the common outcome; trump wins county. accuracy is 1440/1719 = 83.77pct

# checking for multicolinearity
cor(class_train %>% mutate(winner = as.numeric(winner)))
# large correlation between the predictors, will need to try to minimize this using feature engineering


# model specification 
logistic_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# model 1
# fitting/training with tidymodels (chosen off of lower cor levels and predictors I thought were insightful)
class_fit <- logistic_model %>%
  fit(winner ~ total_votes + x0001e + x0002e + x0018e + x0025e + x0026e + x0037e + x0058e + x0086e + x2013_code, data = class_train)


# predicting outcome categories
class_preds <- class_fit %>%
  predict(new_data = class_test,
          type = "class") 
class_prob <- class_fit %>%
  predict(new_data = class_test,
          type = "prob")

# checking model performance w yardstick
class_results <- class_test %>%
  select(winner)%>%
  bind_cols(class_preds, class_prob)

# making a confusion matrix
conf_mat(class_results,
         truth = winner,
         estimate = .pred_class)
# accuracy: classification correctly identified
accuracy(class_results,
         truth = winner,
         estimate = .pred_class)
# specificity: proport of all correct Trump cases
spec(class_results,
    truth = winner,
    estimate = .pred_class)
# sensitivity: proport of all correct Biden cases
sens(class_results,
     truth = winner,
     estimate = .pred_class)

#making this into a tibble 
custom_metrics <- metric_set(accuracy, sens, spec, mcc)
custom_metrics(class_results,
               truth = winner,
               estimate = .pred_class)
# false pos. rate: 1 - spec = 0.027

# plotting the confusion matrix to visualize the data 
conf_mat(class_results,
         truth = winner,
         estimate = .pred_class) %>%
  autoplot(type = "mosaic")

# plotting ROC and finding AUC
class_results %>%
  roc_curve(truth = winner, .pred_Biden) %>%
  autoplot()
roc_auc(class_results,
        truth = winner,
        .pred_Biden)
# ROC AUC: 82.5  

# Automating the workflow using lastfit()
#class_last_fit <- logistic_model %>%
# last_fit(as.factor(winner) ~ x0001e + x0002e + x0018e + x0025e + x0037e + x0058e, split = class_split)
#
#class_last_fit %>%
#  collect_metrics()
#
#class_last_results1 <- class_last_fit %>%
#  collect_predictions()

test_set <- read_csv("test_class.csv")
class_preds_test <- class_fit %>%
  predict(new_data = test_set,
          type = "class")

class_predictions <- test_set %>% select(id) %>%
  bind_cols(class_preds_test)
write_csv(class_predictions, "logistic regression1_2 preds.csv")


# model 2
# feature engineering
class_recipe <- recipe(winner ~.,
                       data = class_train %>%
                         select(-id)) %>%
  # taking log of total votes, etc. to compress range and reduce variability
  step_log(total_votes, x0001e, base = 10) %>%
  #limiting correlation to low (tried at .70, .50, .35, .20)
  step_corr(all_numeric(), threshold = 0.90) %>%
  # centering and scaling
  step_normalize(all_numeric())
  
#class_recipe %>%
#  summary()

# training and transforming recipe
class_recipe_prep <- class_recipe %>%
  prep(training = class_train)
class_recipe_prep2 <- class_recipe_prep %>%
  bake(new_data = NULL)
class_recipe_bake <- class_recipe_prep %>%
  bake(new_data = class_test)

class_fit2 <- logistic_model %>%
  fit(winner ~ .,
      data = class_recipe_prep2)

class_preds2 <- predict(class_fit2,
                        new_data = class_recipe_bake,
                        type = "class")
class_prob2 <- predict(class_fit2,
                        new_data = class_recipe_bake,
                        type = "prob")
class_results2 <- class_test %>%
  select(winner)%>%
  bind_cols(class_preds2, class_prob2)

# model eval
class_results2 %>%
  conf_mat(truth = winner,
           estimate = .pred_class)

custom_metrics(class_results2,
               truth = winner,
               estimate = .pred_class)
# false pos. rate: 1 - spec = 0.04

# plotting the confusion matrix to visualize the data 
conf_mat(class_results2,
         truth = winner,
         estimate = .pred_class) %>%
  autoplot(type = "mosaic")

# plotting ROC and finding AUC
class_results2 %>%
  roc_curve(truth = winner, .pred_Biden) %>%
  autoplot()
roc_auc(class_results2,
        truth = winner,
        .pred_Biden)
# ROC AUC: 71.5  
  
class_preds_test2 <- class_fit2 %>%
  predict(new_data = test_set,
          type = "class")

class_predictions2 <- test_set %>% select(id) %>%
  bind_cols(class_preds_test2)

write_csv(class_predictions2, "logistic regression2_1 preds.csv")
  
  

#class_fit %>% tidy()
# shows int -1.64 and slope estimates stating increasing total_votes by one is associated with dec/inc of log odds by each slope amt
# with trump being the 1 and biden being 2, more votes means probability of biden winning goes up 
# The summary table of the logistic regression fit also includes a z-statistic and p-value which indicates significance
# the function(s) below showcases this phenomenon by adding a 0 to total votes
# predict(class_fit, new_data = data.frame(total_votes = 52346), type = "prob")
# predict(class_fit, new_data = data.frame(total_votes = 523460), type = "prob")


# F-score
f_meas(class_preds, truth = Truth, estimate = .pred_class, event_level = "first") #Trump
f_meas(class_preds, truth = Truth, estimate = .pred_class, event_level = "second") #Biden
