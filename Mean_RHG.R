# Load required libraries
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("Metrics", quietly = TRUE)) install.packages("Metrics")

library(randomForest)
library(caret)
library(dplyr)
library(ggplot2)
library(Metrics)

# Load data for district RHG
input_csv_RHG <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/RHG_data.csv"

data_RHG <- read.csv(input_csv_RHG)


# Clean and preprocess the data
data_cleaned_RHG <- data_RHG %>%
  select(-c(SD_Biomass, GRID_ID)) %>%  # Remove SD_Biomass and GRID_ID columns
  na.omit()                            # Remove rows with missing values

# Separate features and target
target_var_RHG <- "Mean_Biomass"
predictors_RHG <- setdiff(names(data_cleaned_RHG), target_var_RHG)

# Prepare feature matrix (x) and target (y)
x_mean_RHG <- data_cleaned_RHG[, predictors_RHG]
y_mean_RHG <- data_cleaned_RHG[[target_var_RHG]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_RHG <- createDataPartition(y_mean_RHG, p = 0.8, list = FALSE)
x_train_mean_RHG <- x_mean_RHG[train_indices_RHG, ]
y_train_mean_RHG <- y_mean_RHG[train_indices_RHG]
x_test_mean_RHG <- x_mean_RHG[-train_indices_RHG, ]
y_test_mean_RHG <- y_mean_RHG[-train_indices_RHG]

# Train the simple Random Forest model (no CV)
set.seed(52)
rf_model_mean_RHG <- randomForest(
  x = x_train_mean_RHG,
  y = y_train_mean_RHG,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_mean_RHG)

# Extract and normalize variable importance
var_imp_RHG <- data.frame(
  variable = rownames(rf_model_mean_RHG$importance),
  Overall = rf_model_mean_RHG$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot normalized variable importance
ggplot(var_imp_RHG, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6, color = "black", linewidth = 0.3) +
  labs(
    x = "Variables",
    y = "Variable Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 12, angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14, color = "black")
  )

# Make predictions on training and test sets
y_pred_train_mean_RHG <- predict(rf_model_mean_RHG, x_train_mean_RHG)
y_pred_mean_RHG <- predict(rf_model_mean_RHG, x_test_mean_RHG)

# Evaluate training performance
r2_train_mean_RHG <- caret::R2(y_pred_train_mean_RHG, y_train_mean_RHG)
rmse_train_mean_RHG <- Metrics::rmse(y_train_mean_RHG, y_pred_train_mean_RHG)
mae_train_mean_RHG <- Metrics::mae(y_train_mean_RHG, y_pred_train_mean_RHG)

# Evaluate test performance
r2_test_mean_RHG <- caret::R2(y_pred_mean_RHG, y_test_mean_RHG)
rmse_test_mean_RHG <- Metrics::rmse(y_test_mean_RHG, y_pred_mean_RHG)
mae_test_mean_RHG <- Metrics::mae(y_test_mean_RHG, y_pred_mean_RHG)

# Print training and test metrics
cat(sprintf(
  "\nMean biomass RHG - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_mean_RHG, rmse_train_mean_RHG, mae_train_mean_RHG
))
cat(sprintf(
  "Mean biomass RHG - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_mean_RHG, rmse_test_mean_RHG, mae_test_mean_RHG
))

# Validation plot for Mean_Biomass (RHG district)
validation_data_mean_RHG <- data.frame(
  Actual = y_test_mean_RHG,
  Predicted = y_pred_mean_RHG
)

# Calculate axis limits and breaks dynamically
step_size <- 100
axis_limits <- c(
  min(validation_data_mean_RHG$Actual, validation_data_mean_RHG$Predicted) - 10,
  max(800,max(validation_data_mean_RHG$Actual, validation_data_mean_RHG$Predicted) + 10)
)
breaks <- seq(
  floor(axis_limits[1] / step_size) * step_size,
  ceiling(axis_limits[2] / step_size) * step_size,
  by = step_size
)

ggplot(validation_data_mean_RHG, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1, xlim = axis_limits, ylim = axis_limits) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual Mean Biomass RHG (g/m²)",
    y = "Predicted Mean Biomass RHG (g/m²)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.title = element_text(size = 14, color = "black"),
    axis.text = element_text(size = 14, color = "black"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "white", color = "white"),
    plot.margin = margin(5, 5, 5, 30)
  ) +
  annotate("text", 
           x = axis_limits[1] + 10,  
           y = axis_limits[2] - 10,  
           label = "(a)", fontface = "bold", size = 6, color = "black", hjust = 0) +
  annotate("text", 
           x = axis_limits[2] - 5,  
           y = axis_limits[1] + 50,   
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_mean_RHG, rmse_test_mean_RHG, mae_test_mean_RHG),
           hjust = 1, size = 5, color = "black")

# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_mean_RHG.png", 
       width = 15, height = 15, units ='cm', dpi = 300)
