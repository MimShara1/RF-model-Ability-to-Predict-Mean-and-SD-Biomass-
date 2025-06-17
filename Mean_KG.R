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

# Load data for district KG
input_csv <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/KG_data.csv"
data_KG <- read.csv(input_csv)

# Clean and preprocess the data
data_cleaned_KG <- data_KG %>%
  select(-c(SD_Biomass, GRID_ID)) %>%   # Remove SD_Biomass and GRID_ID columns
  na.omit()                            # Remove rows with missing values

# Separate features and target
target_var_KG <- "Mean_Biomass"
predictors_KG <- setdiff(names(data_cleaned_KG), target_var_KG)

# Prepare feature matrix (x) and target (y)
x_mean_KG <- data_cleaned_KG[, predictors_KG]
y_mean_KG <- data_cleaned_KG[[target_var_KG]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_KG <- createDataPartition(y_mean_KG, p = 0.8, list = FALSE)
x_train_mean_KG <- x_mean_KG[train_indices_KG, ]
y_train_mean_KG <- y_mean_KG[train_indices_KG]
x_test_mean_KG <- x_mean_KG[-train_indices_KG, ]
y_test_mean_KG <- y_mean_KG[-train_indices_KG]

# Train the simple Random Forest model 
set.seed(52)
rf_model_mean_KG <- randomForest(
  x = x_train_mean_KG,
  y = y_train_mean_KG,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_mean_KG)

# Extract and normalize variable importance
var_imp_KG <- data.frame(
  variable = rownames(rf_model_mean_KG$importance),
  Overall = rf_model_mean_KG$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot normalized variable importance
ggplot(var_imp_KG, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
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
y_pred_train_mean_KG <- predict(rf_model_mean_KG, x_train_mean_KG)
y_pred_mean_KG <- predict(rf_model_mean_KG, x_test_mean_KG)

# Evaluate training performance
r2_train_mean_KG <- caret::R2(y_pred_train_mean_KG, y_train_mean_KG)
rmse_train_mean_KG <- Metrics::rmse(y_train_mean_KG, y_pred_train_mean_KG)
mae_train_mean_KG <- Metrics::mae(y_train_mean_KG, y_pred_train_mean_KG)

# Evaluate test performance
r2_test_mean_KG <- caret::R2(y_pred_mean_KG, y_test_mean_KG)
rmse_test_mean_KG <- Metrics::rmse(y_test_mean_KG, y_pred_mean_KG)
mae_test_mean_KG <- Metrics::mae(y_test_mean_KG, y_pred_mean_KG)

# Print training and test metrics
cat(sprintf(
  "\nMean biomass KG - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_mean_KG, rmse_train_mean_KG, mae_train_mean_KG
))
cat(sprintf(
  "Mean biomass KG - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_mean_KG, rmse_test_mean_KG, mae_test_mean_KG
))

# Validation Plot for Mean_Biomass (KG district)

validation_data_mean_KG <- data.frame(
  Actual = y_test_mean_KG,
  Predicted = y_pred_mean_KG
)

# Calculate axis limits and breaks dynamically
step_size <- 100
axis_limits <- c(
  min(validation_data_mean_KG$Actual, validation_data_mean_KG$Predicted) - 10,
  max(900, max(validation_data_mean_KG$Actual, validation_data_mean_KG$Predicted) + 10)
)

breaks <- seq(
  floor(axis_limits[1] / step_size) * step_size,
  ceiling(axis_limits[2] / step_size) * step_size,
  by = step_size
)

ggplot(validation_data_mean_KG, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1, xlim = axis_limits, ylim = axis_limits) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual Mean Biomass KG (g/m²)",
    y = "Predicted Mean Biomass KG (g/m²)"
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
           label = "(b)", fontface = "bold", size = 6, color = "black", hjust = 0) +
  annotate("text", 
           x = axis_limits[2] - 5,  
           y = axis_limits[1] + 45,   
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_mean_KG, rmse_test_mean_KG, mae_test_mean_KG),
           hjust = 1, size = 5, color = "black")
# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_mean_KG.png", 
       width = 15, height = 15, units ='cm', dpi = 300)
