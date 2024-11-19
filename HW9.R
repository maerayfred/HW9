library(tidyverse)
library(tidymodels)
set.seed(10)
bike_data <- read_csv("https://www4.stat.ncsu.edu/~online/datasets/bikeDetails.csv")
bike_data <- bike_data |> 
  mutate(log_selling_price = log(selling_price), 
         log_km_driven = log(km_driven),
         owners = ifelse(owner == "1st owner", "single", "multiple")) |>
  select(log_km_driven, log_selling_price, everything())
#use tidymodel functions for splitting the data
bike_split <- initial_split(bike_data, prop = 0.7)
bike_train <- training(bike_split)
bike_test <- testing(bike_split)


#create folds
bike_CV_folds <- vfold_cv(bike_train, 10)

#set up how we'll fit our linear model
MLR_spec <- linear_reg() |>
  set_engine("lm")


#define our MLR models
MLR_recipe1 <- recipe(log_selling_price ~ log_km_driven + owners + year, 
                      data = bike_train) |>
  step_dummy(owners)
MLR_recipe2 <- recipe(log_selling_price ~ log_km_driven + owners,
                      data = bike_train) |>
  step_dummy(owners) |>
  step_interact(~log_km_driven:starts_with("owner"))

MLR_recipe3 <- recipe(log_selling_price ~ log_km_driven + owners + year,
                      data = bike_train) |>
  step_dummy(owners) |>
  step_interact(~log_km_driven:starts_with("owner") + log_km_driven:year + starts_with("owner"):year)

MLR_wkf1 <- workflow() |>
  add_recipe(MLR_recipe1) |>
  add_model(MLR_spec)

MLR_wkf2 <- workflow() |>
  add_recipe(MLR_recipe2) |>
  add_model(MLR_spec)

MLR_wkf3 <- workflow() |>
  add_recipe(MLR_recipe3) |>
  add_model(MLR_spec)

MLR_fit1 <-  MLR_wkf1 |>
  fit_resamples(bike_CV_folds)

MLR_fit2 <- MLR_wkf2 |>
  fit_resamples(bike_CV_folds) 

MLR_fit3 <- MLR_wkf3 |>
  fit_resamples(bike_CV_folds)

rbind(MLR_fit1 |> collect_metrics() |> filter(.metric == "rmse"),
      MLR_fit2 |> collect_metrics() |> filter(.metric == "rmse"),
      MLR_fit3 |> collect_metrics() |> filter(.metric == "rmse")) |> 
  mutate(Model = c("Model 1", "Model 2", "Model 3")) |>
  select(Model, mean, n, std_err)

MLR_final <-  MLR_wkf3 |>
  fit(bike_train)
tidy(MLR_final)


#set up how we'll fit our LASSO model
#code modified from https://juliasilge.com/blog/lasso-the-office/
LASSO_recipe <- recipe(log_selling_price ~ log_km_driven + owners + year, 
                       data = bike_train) |>
  step_dummy(owners) |>
  step_normalize(log_km_driven, year)

LASSO_spec <- linear_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet")

LASSO_wkf <- workflow() |>
  add_recipe(LASSO_recipe) |>
  add_model(LASSO_spec)
LASSO_wkf


library(glmnet)

#A warning will occur for one value of the tuning parameter, safe to ignore
LASSO_grid <- LASSO_wkf |>
  tune_grid(resamples = bike_CV_folds,
            grid = grid_regular(penalty(), levels = 200)) 

LASSO_grid

LASSO_grid[1, ".metrics"][[1]]

LASSO_grid |>
  collect_metrics() |>
  filter(.metric == "rmse")

LASSO_grid |>
  collect_metrics() |>
  filter(.metric == "rmse") |>
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_line()


lowest_rmse <- LASSO_grid |>
  select_best(metric = "rmse")
lowest_rmse


LASSO_wkf |>
  finalize_workflow(lowest_rmse)

#fit it to the entire training set to see the model fit
LASSO_final <- LASSO_wkf |>
  finalize_workflow(lowest_rmse) |>
  fit(bike_train)
tidy(LASSO_final)

MLR_wkf3 |>
  last_fit(bike_split) |>
  collect_metrics()

LASSO_wkf |>
  finalize_workflow(lowest_rmse) |>
  last_fit(bike_split) |>
  collect_metrics()


MLR_final |>
  predict(bike_test) |>
  pull() |>
  rmse_vec(truth = bike_test$log_selling_price)

LASSO_final |>
  predict(bike_test) |>
  pull() |>
  rmse_vec(truth = bike_test$log_selling_price)


final_model <- MLR_wkf3 |>
  fit(bike_data) 
tidy(final_model)

almost_usual_fit <- extract_fit_parsnip(final_model)
usual_fit <- almost_usual_fit$fit
summary(usual_fit)


