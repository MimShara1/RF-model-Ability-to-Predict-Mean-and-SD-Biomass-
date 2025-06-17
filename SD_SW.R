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

# Load data for district SW
input_csv_sw <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/SW_data.csv"
data_SW <- read.csv(input_csv_sw)

# Clean and preprocess the data
data_cleaned_SW <- data_SW %>%
  select(-c(Mean_Biomass, GRID_ID)) %>%  # Remove Mean_Biomass and GRID_ID columns
  na.omit()  # Remove N/A values

# Separate features and target
target_var_SW <- "SD_Biomass"
predictors_SW <- setdiff(names(data_cleaned_SW), target_var_SW)

# Prepare feature matrix (x) and target (y)
x_sd_SW <- data_cleaned_SW[, predictors_SW]
y_sd_SW <- data_cleaned_SW[[target_var_SW]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_SW <- createDataPartition(y_sd_SW, p = 0.8, list = FALSE)
x_train_sd_SW <- x_sd_SW[train_indices_SW, ]
y_train_sd_SW <- y_sd_SW[train_indices_SW]
x_test_sd_SW <- x_sd_SW[-train_indices_SW, ]
y_test_sd_SW <- y_sd_SW[-train_indices_SW]

# Train Random Forest model
set.seed(52)
rf_model_sd_SW <- randomForest(
  x = x_train_sd_SW,
  y = y_train_sd_SW,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_sd_SW)

# Extract and normalize variable importance
var_imp_SW <- data.frame(
  variable = rownames(rf_model_sd_SW$importance),
  Overall = rf_model_sd_SW$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot variable importance
ggplot(var_imp_SW, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6, color = "black", linewidth = 0.3) +
  labs(x = "Variables", y = "Variable Importance") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 12, angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14, color = "black")
  )

# Predictions
y_pred_train_sd_SW <- predict(rf_model_sd_SW, x_train_sd_SW)
y_pred_sd_SW <- predict(rf_model_sd_SW, x_test_sd_SW)

# Metrics
r2_train_sd_SW <- caret::R2(y_pred_train_sd_SW, y_train_sd_SW)
rmse_train_sd_SW <- Metrics::rmse(y_train_sd_SW, y_pred_train_sd_SW)
mae_train_sd_SW <- Metrics::mae(y_train_sd_SW, y_pred_train_sd_SW)

r2_test_sd_SW <- caret::R2(y_pred_sd_SW, y_test_sd_SW)
rmse_test_sd_SW <- Metrics::rmse(y_test_sd_SW, y_pred_sd_SW)
mae_test_sd_SW <- Metrics::mae(y_test_sd_SW, y_pred_sd_SW)

cat(sprintf(
  "\nSD biomass SW - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_sd_SW, rmse_train_sd_SW, mae_train_sd_SW
))
cat(sprintf(
  "SD biomass SW - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_sd_SW, rmse_test_sd_SW, mae_test_sd_SW
))

# Validation Plot (Fixed Axis)
validation_data_sd_SW <- data.frame(
  Actual = y_test_sd_SW,
  Predicted = y_pred_sd_SW
)

# Fixed axis for consistency across districts
axis_limits <- c(0, 400)
breaks <- seq(0, 400, 100)

ggplot(validation_data_sd_SW, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual SD Biomass SW (g/m²)",
    y = "Predicted SD Biomass SW (g/m²)"
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
           label = "(d)", fontface = "bold", size = 6, color = "black", hjust = 0) +
  annotate("text", 
           x = axis_limits[2] - 10,
           y = axis_limits[1] + 50,
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_sd_SW, rmse_test_sd_SW, mae_test_sd_SW),
           hjust = 1, size = 5, color = "black")
# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_SD_SW.png", 
       width = 15, height = 15, units ='cm', dpi = 300)
