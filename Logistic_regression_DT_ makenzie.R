library(tidyverse)
library(tidymodels)
library(ISLR)


classification <- read_csv("train_class.csv")
col_explain <- read_csv("col_descriptions.csv")

# EDA and cleaning the data
#looking at the data set
summary(classification)
#NA's present nearing the end of the data set betw. cols pertaining to income and GDP
class_clean <- classification %>%
  filter(complete.cases(.)) #removes incomplete rows with NA values
# removing any duplicated columns
class_clean <- class_clean %>%
  select(-c(name, x0021e, x0024e, x0033e, x0034e, x0035e)) %>%
  mutate(winner = as.factor(winner))
glimpse(class_clean)

#visualizing the data 
ggplot(class_clean, aes(x= winner)) +
  geom_histogram(stat = "count") +
  labs(x= "winner", y= "count")
# finding total winnings
table(class_clean$winner)
# Biden won 373 total counties, Trump 1921; indicates that the counties Biden won had high electoral votes
ggplot(class_clean, aes(x= total_votes, y= winner)) +
  geom_boxplot() +
  labs(x= "total votes", y= "winner")
# The box plot shows that, as expected, the counties Biden one in had a higher vote count 
# meaning a higher county population and thus they have higher electoral power

# summaries of mean voters amount
class_clean %>%
  group_by(winner) %>%
  summarise(mean = mean(total_votes),
            sd = sd(total_votes), 
            n = n())
# this shows that for Biden's 373 county wins, he had an avg of 171,768 voters and Trump's 1921 county
# wins only had an avg of 25,193 voters. This showcases the power Biden's counties had in the electoral process

# data resampling
set.seed(16)
class_split <- initial_split(class_clean, prop = 0.75, strata = winner)
class_train <- training(class_split)
class_test <- testing(class_split)

# a little EDA on the split training set
table(class_train$winner)
# Biden won 279 counties, Trump 1440
# baseline predicts the common outcome; trump wins county. accuracy is 1440/1719 = 83.77pct

# checking for multicolinearity
class_train %>% select_if(is.numeric) %>% cor()
# large correlation between the predictors, will need to try to minimize this using feature engineering


# model specification 
logistic_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# model 1: simple logistic regression
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

#making this into a tibble and ROC AUC
custom_metrics <- metric_set(accuracy, sens, spec, roc_auc)
custom_metrics(class_results,
               truth = winner,
               estimate = .pred_class,
               .pred_Biden)
# false pos. rate: 1 - spec = 0.027
# ROC AUC: 82.5 

# plotting the confusion matrix to visualize the data 
conf_mat(class_results,
         truth = winner,
         estimate = .pred_class) %>%
  autoplot(type = "mosaic")

# plotting ROC
class_results %>%
  roc_curve(truth = winner, .pred_Biden) %>%
  autoplot()

 

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


# model 2: feature engineering logistic regression
# feature engineering
set.seed(16)
class_recipe <- recipe(winner ~.,
                       data = class_train %>%
                         select(-id)) %>%
  # taking log of total votes, etc. to compress range and reduce variability
  step_log(c(total_votes, x0001e,income_per_cap_2016:gdp_2020), base = 10) %>%
  #limiting correlation to low (tried at .70, .50, .35, .20)
  step_corr(all_numeric(), threshold = 0.90) %>%
  # centering and scaling
  step_normalize(all_numeric())

#class_recipe %>%
#  summary()

# training and transforming recipe
set.seed(16)
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
               estimate = .pred_class,
               .pred_Biden)
# false pos. rate: 1 - spec = 0.042
# ROC AUC: 74.6 

# plotting the confusion matrix to visualize the data 
conf_mat(class_results2,
         truth = winner,
         estimate = .pred_class) %>%
  autoplot(type = "mosaic")

# plotting ROC and finding AUC
class_results2 %>%
  roc_curve(truth = winner, .pred_Biden) %>%
  autoplot()
  
class_preds_test2 <- class_fit2 %>%
  predict(new_data = test_set,
          type = "class")

class_predictions2 <- test_set %>% select(id) %>%
  bind_cols(class_preds_test2)

write_csv(class_predictions2, "logistic regression2_2 preds.csv")

# model 3: decision trees
dt_model <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")

#combining models and recipes with workflow 
set.seed(16)
class_workflow <- workflow() %>%
  add_model(dt_model) %>%
  add_recipe(class_recipe)

# training and fitting DT to training data
class_workflow_fit <- class_workflow %>%
  last_fit(split = class_split)

class_workflow_fit %>% collect_metrics()

# collecting predictions
set.seed(16)
class_results3_a <- class_workflow_fit %>%
  collect_predictions()
  
# evaluating performance
class_results3_a %>%
  conf_mat(truth = winner,
           estimate = .pred_class)


custom_metrics(class_results3_a,
               truth = winner,
               estimate = .pred_class, 
               .pred_Biden)
# false pos. rate: 1 - spec = 0.05
# ROC AUC: 79.2

# plotting the confusion matrix to visualize the data 
conf_mat(class_results3_a,
         truth = winner,
         estimate = .pred_class) %>%
  autoplot(type = "mosaic")

# plotting ROC 
class_results3_a %>%
  roc_curve(truth = winner, .pred_Biden) %>%
  autoplot()

# cross validation using vfold_cv() to compare model types
set.seed(16)
class_folds <- vfold_cv(class_train,
                        v = 10,
                        strata = winner)

# training using cross validation
class_resample_fit <- class_workflow %>%
  fit_resamples(resamples = class_folds)

# looking at cv results
class_res_metricss <- class_resample_fit %>%
  collect_metrics(summarize = FALSE)

# summarizing results
class_res_metricss %>%
  group_by(.metric) %>%
  summarise(min = min(.estimate),
            median =median(.estimate),
            max = max(.estimate),
            mean = mean(.estimate),
            sd = sd(.estimate))

# tuning to find the optimal set of hyper parameters
# cost complexity: penalizes large number of nodes
# tree depth: max path allowed root to node
# min n: min data points req in node
dt_tune_model <- decision_tree(cost_complexity = tune(),
                               tree_depth = tune(),
                               min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# creating a tuning workflow
class_tune_workflow <- class_workflow %>%
  update_model(dt_tune_model)
# uses grid search to find optimal combination 
# creating random grid to generate combinations
set.seed(16)
class_grid <- grid_random(parameters(dt_tune_model),
                          size = 5)
class_dt_tune <- class_tune_workflow %>%
  tune_grid(resamples = class_folds,
            grid = class_grid,
            metrics = custom_metrics)
# looking at tuning results
class_dt_metrics <- class_dt_tune %>% collect_metrics(summarize = FALSE)

#summarizing results of each fold
class_dt_metrics %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))

# selecting the best results
class_dt_tune %>%
  show_best(metric = "roc_auc", n = 5)

best_dt_model <- class_dt_tune %>%
  select_best(metric = "roc_auc")

final_class_workflow <- class_tune_workflow %>%
  finalize_workflow(best_dt_model)

# training and fitting DT to training data
class_final_fit <- final_class_workflow %>%
  last_fit(split = class_split)

class_final_fit %>% collect_metrics()

# collecting test predictions and plotting ROC curve
  class_results3_b <- class_final_fit %>%
  collect_predictions()

  class_results3_b %>%
    roc_curve(truth = winner , .pred_Biden) %>%
    autoplot()
  
  class_preds_test3 <- class_final_fit %>%
    extract_workflow() %>%
    predict(new_data = test_set,
            type = "class")
  
  class_predictions3 <- test_set %>% select(id) %>%
    bind_cols(class_preds_test3)
  
  write_csv(class_predictions3, "decision tree preds.csv")
  




