# Load required libraries
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("Metrics", quietly = TRUE)) install.packages("Metrics")

library(randomForest)
library(dplyr)
library(ggplot2)
library(Metrics)

# Load data
input_csv <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/Hexagon_Mean_SD_Predictors.csv"
data_reg <- read.csv(input_csv)

# Clean and preprocess the data for SD_Biomass prediction
data_cleaned_reg <- data_reg %>%
  select(-c(Mean_Biomass, GRID_ID)) %>%  # Remove Mean_Biomass and GRID_ID columns
  na.omit()                             # Remove rows with NA values

# Separate features and target
target_var_reg <- "SD_Biomass"
predictors_reg <- setdiff(names(data_cleaned_reg), target_var_reg)

# Prepare feature matrix (x) and target (y)
x_sd_reg <- data_cleaned_reg[, predictors_reg]
y_sd_reg <- data_cleaned_reg[[target_var_reg]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_reg <- caret::createDataPartition(y_sd_reg, p = 0.8, list = FALSE)
x_train_sd_reg <- x_sd_reg[train_indices_reg, ]
y_train_sd_reg <- y_sd_reg[train_indices_reg]
x_test_sd_reg <- x_sd_reg[-train_indices_reg, ]
y_test_sd_reg <- y_sd_reg[-train_indices_reg]

# Train the Random Forest model
set.seed(52)
rf_model_sd <- randomForest(
  x = x_train_sd_reg,
  y = y_train_sd_reg,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_sd)

# Extract and normalize variable importance
var_imp_sd_reg <- data.frame(
  variable = rownames(rf_model_sd$importance),
  Overall = rf_model_sd$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot normalized variable importance
ggplot(var_imp_sd_reg, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
  geom_bar(stat = "identity", fill = "brown", width = 0.6, color = "black", linewidth = 0.3) +
  labs(
    x = "Variables",
    y = "Variable Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 14, color = 'black'),
    axis.text.y = element_text(size = 14, color = 'black'),
    axis.title = element_text(size = 14, color = 'black')
  )+
  coord_flip()

# Save the variable importance plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/var_imp_reg_sd.png", 
       width = 15, height = 15, units = 'cm', dpi = 300)

# Predictions on train and test sets
y_pred_train_sd_reg <- predict(rf_model_sd, x_train_sd_reg)
y_pred_test_sd_reg <- predict(rf_model_sd, x_test_sd_reg)

# Compute evaluation metrics
r2_train_sd <- caret::R2(y_pred_train_sd_reg, y_train_sd_reg)
rmse_train_sd <- Metrics::rmse(y_train_sd_reg, y_pred_train_sd_reg)
mae_train_sd <- Metrics::mae(y_train_sd_reg, y_pred_train_sd_reg)

r2_test_sd <- caret::R2(y_pred_test_sd_reg, y_test_sd_reg)
rmse_test_sd <- Metrics::rmse(y_test_sd_reg, y_pred_test_sd_reg)
mae_test_sd <- Metrics::mae(y_test_sd_reg, y_pred_test_sd_reg)

# Print training and test performance
cat(sprintf(
  "SD biomass Rhön region - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_sd, rmse_train_sd, mae_train_sd
))
cat(sprintf(
  "SD biomass Rhön region - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_sd, rmse_test_sd, mae_test_sd
))

# Prepare data for validation plot
validation_data_sd_reg <- data.frame(
  Actual = y_test_sd_reg,
  Predicted = y_pred_test_sd_reg
)

# Axis settings for validation plot
step_size <- 100
axis_limits <- c(
  min(validation_data_sd_reg$Actual, validation_data_sd_reg$Predicted) - 10,
  max(400, max(validation_data_sd_reg$Actual, validation_data_sd_reg$Predicted) + 10)
)
breaks <- seq(
  floor(axis_limits[1] / step_size) * step_size,
  ceiling(axis_limits[2] / step_size) * step_size,
  by = step_size
)

# Validation plot
ggplot(validation_data_sd_reg, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1, xlim = axis_limits, ylim = axis_limits) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual SD Biomass (g/m²)",
    y = "Predicted SD Biomass (g/m²)"
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
           label = "(b)", fontface = "bold", size = 5, color = "black", hjust = 0) +
  annotate("text", 
           x = axis_limits[2] - 15,  
           y = axis_limits[1] + 50,   
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_sd, rmse_test_sd, mae_test_sd),
           hjust = 1, size = 5, color = "black")

# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_reg_sd.png", 
       width = 15, height = 15, units ='cm', dpi = 300)
