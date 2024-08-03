library(kknn)
library(tidyverse)
library(workflowsets)
library(tidymodels)

classification <- read_csv("train_class.csv")

# removing convoluted data to refine 
class_clean <- classification %>%
  filter(complete.cases(.)) %>%
  select(-c(name, x0001e, x0019e:x0025e, x0029e,  x0033e, x0034e:x0036e, x0058e, x0076e, x0087e, 
            c01_001e, c01_006e, c01_014e:c01_016e, c01_019e, c01_022e, c01_025e)) %>%
  mutate(winner = as.factor(winner))


set.seed(16)
class_split <- initial_split(class_clean, prop = 0.80, strata = winner)
class_train <- training(class_split)
class_test <- testing(class_split)

# using PCA to de-correlate data
class_recipe <-
  recipe(winner ~ ., data = class_train) %>%
  step_normalize(all_predictors())
filter_recipe <- class_recipe %>%
  step_corr(all_predictors(), threshold = tune())
pca_recipe <- class_recipe %>%
  step_pca(all_numeric_predictors(), num_comp = tune()) %>%
  step_normalize(all_predictors())

# Assessing the logistic regression, decision trees, and KNN
# finding optimal values with the training set using tune() 
logReg_spec <-
  logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")
  
dt_spec <-
  decision_tree(cost_complexity = tune(),
                tree_depth = tune(), 
                min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

knn_spec <-
  nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")


chi_models <-
  workflow_set(
    preproc = list(simple = class_recipe, filter = filter_recipe,
                   pca = pca_recipe),
    models = list(glm = logReg_spec, dt = dt_spec,
                  knn = knn_spec),
    cross = TRUE
  )

train_folds <- vfold_cv(class_train)

set.seed(16)
chi_models <-
  chi_models %>%
  workflow_map("tune_grid", resamples = train_folds, grid = 10, verbose = TRUE)

autoplot(chi_models)
autoplot(chi_models, select_best = TRUE)

chi_results <- rank_results(chi_models, rank_metric = "roc_auc", select_best = TRUE) %>%
  select(rank, mean, model, wflow_id, .config)
print(chi_results)

# Because of the results I worked on a decision tree model instead of KNN
