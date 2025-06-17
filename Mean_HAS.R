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

# Load data for district HAS
input_csv_HAS <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/HAS_data.csv"
data_HAS <- read.csv(input_csv_HAS)

# Clean and preprocess the data
data_cleaned_HAS <- data_HAS %>%
  select(-c(SD_Biomass, GRID_ID)) %>%  # Remove SD_Biomass and GRID_ID columns
  na.omit()                            # Remove rows with missing values

# Separate features and target
target_var_HAS <- "Mean_Biomass"
predictors_HAS <- setdiff(names(data_cleaned_HAS), target_var_HAS)

# Prepare feature matrix (x) and target (y)
x_mean_HAS <- data_cleaned_HAS[, predictors_HAS]
y_mean_HAS <- data_cleaned_HAS[[target_var_HAS]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_HAS <- createDataPartition(y_mean_HAS, p = 0.8, list = FALSE)
x_train_mean_HAS <- x_mean_HAS[train_indices_HAS, ]
y_train_mean_HAS <- y_mean_HAS[train_indices_HAS]
x_test_mean_HAS <- x_mean_HAS[-train_indices_HAS, ]
y_test_mean_HAS <- y_mean_HAS[-train_indices_HAS]

# Train the Random Forest model (no CV)
set.seed(52)
rf_model_mean_HAS <- randomForest(
  x = x_train_mean_HAS,
  y = y_train_mean_HAS,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_mean_HAS)

# Extract and normalize variable importance (IncNodePurity)
var_imp_HAS <- data.frame(
  variable = rownames(rf_model_mean_HAS$importance),
  Overall = rf_model_mean_HAS$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot normalized variable importance
ggplot(var_imp_HAS, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6, color = "black", linewidth = 0.3) +
  labs(
    x = "Variables",
    y = "Variable Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 14, angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 14, color = "black"),
    axis.title = element_text(size = 14, color = "black")
  )

# Make predictions on train and test sets
y_pred_train_mean_HAS <- predict(rf_model_mean_HAS, x_train_mean_HAS)
y_pred_mean_HAS <- predict(rf_model_mean_HAS, x_test_mean_HAS)

# Evaluate training performance
r2_train_mean_HAS <- caret::R2(y_pred_train_mean_HAS, y_train_mean_HAS)
rmse_train_mean_HAS <- Metrics::rmse(y_train_mean_HAS, y_pred_train_mean_HAS)
mae_train_mean_HAS <- Metrics::mae(y_train_mean_HAS, y_pred_train_mean_HAS)

# Evaluate test performance
r2_test_mean_HAS <- caret::R2(y_pred_mean_HAS, y_test_mean_HAS)
rmse_test_mean_HAS <- Metrics::rmse(y_test_mean_HAS, y_pred_mean_HAS)
mae_test_mean_HAS <- Metrics::mae(y_test_mean_HAS, y_pred_mean_HAS)

# Print training and test metrics
cat(sprintf(
  "\nMean biomass HAS - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_mean_HAS, rmse_train_mean_HAS, mae_train_mean_HAS
))
cat(sprintf(
  "Mean biomass HAS - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_mean_HAS, rmse_test_mean_HAS, mae_test_mean_HAS
))

# Validation plot for Mean_Biomass (HAS district)
validation_data_mean_HAS <- data.frame(
  Actual = y_test_mean_HAS,
  Predicted = y_pred_mean_HAS
)

# Calculate axis limits and breaks dynamically
step_size <- 100
axis_limits <- c(
  min(validation_data_mean_HAS$Actual, validation_data_mean_HAS$Predicted) - 10,
  max(900, max(validation_data_mean_HAS$Actual, validation_data_mean_HAS$Predicted) + 10)
)
breaks <- seq(
  floor(axis_limits[1] / step_size) * step_size,
  ceiling(axis_limits[2] / step_size) * step_size,
  by = step_size
)

ggplot(validation_data_mean_HAS, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1, xlim = axis_limits, ylim = axis_limits) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual Mean Biomass HAS (g/m²)",
    y = "Predicted Mean Biomass HAS (g/m²)"
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
           label = "(c)", fontface = "bold", size = 6, color = "black", hjust = 0) +
  annotate("text", 
           x = axis_limits[2] - 5,  
           y = axis_limits[1] + 50,   
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_mean_HAS, rmse_test_mean_HAS, mae_test_mean_HAS),
           hjust = 1, size = 5, color = "black")

# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_mean_HAS.png", 
       width = 15, height = 15, units ='cm', dpi = 300)