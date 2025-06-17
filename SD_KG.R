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
  select(-c(Mean_Biomass, GRID_ID)) %>%  # Remove Mean_Biomass and GRID_ID columns
  na.omit()  #Remove N/A values

# Separate features and target
target_var_KG <- "SD_Biomass"
predictors_KG <- setdiff(names(data_cleaned_KG), target_var_KG)

# Prepare feature matrix (x) and target (y)
x_sd_KG <- data_cleaned_KG[, predictors_KG]
y_sd_KG <- data_cleaned_KG[[target_var_KG]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_KG <- createDataPartition(y_sd_KG, p = 0.8, list = FALSE)
x_train_sd_KG <- x_sd_KG[train_indices_KG, ]
y_train_sd_KG <- y_sd_KG[train_indices_KG]
x_test_sd_KG <- x_sd_KG[-train_indices_KG, ]
y_test_sd_KG <- y_sd_KG[-train_indices_KG]

# Train Random Forest model
set.seed(52)
rf_model_sd_KG <- randomForest(
  x = x_train_sd_KG,
  y = y_train_sd_KG,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_sd_KG)

# Extract and normalize variable importance
var_imp_KG <- data.frame(
  variable = rownames(rf_model_sd_KG$importance),
  Overall = rf_model_sd_KG$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot variable importance
ggplot(var_imp_KG, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6, color = "black", linewidth = 0.3) +
  labs(x = "Variables", y = "Variable Importance") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 12, angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14, color = "black")
  )

# Predictions
y_pred_train_sd_KG <- predict(rf_model_sd_KG, x_train_sd_KG)
y_pred_sd_KG <- predict(rf_model_sd_KG, x_test_sd_KG)

# Metrics
r2_train_sd_KG <- caret::R2(y_pred_train_sd_KG, y_train_sd_KG)
rmse_train_sd_KG <- Metrics::rmse(y_train_sd_KG, y_pred_train_sd_KG)
mae_train_sd_KG <- Metrics::mae(y_train_sd_KG, y_pred_train_sd_KG)

r2_test_sd_KG <- caret::R2(y_pred_sd_KG, y_test_sd_KG)
rmse_test_sd_KG <- Metrics::rmse(y_test_sd_KG, y_pred_sd_KG)
mae_test_sd_KG <- Metrics::mae(y_test_sd_KG, y_pred_sd_KG)

cat(sprintf(
  "\nSD biomass KG - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_sd_KG, rmse_train_sd_KG, mae_train_sd_KG
))
cat(sprintf(
  "SD biomass KG - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_sd_KG, rmse_test_sd_KG, mae_test_sd_KG
))

# Validation Plot (Fixed Axis)
validation_data_sd_KG <- data.frame(
  Actual = y_test_sd_KG,
  Predicted = y_pred_sd_KG
)

# Fixed axis for consistency across districts
axis_limits <- c(0, 400)
breaks <- seq(0, 400, 100)

ggplot(validation_data_sd_KG, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual SD Biomass KG (g/m²)",
    y = "Predicted SD Biomass KG (g/m²)"
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
           x = axis_limits[2] - 10,
           y = axis_limits[1] + 50,
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_sd_KG, rmse_test_sd_KG, mae_test_sd_KG),
           hjust = 1, size = 5, color = "black")
# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_SD_KG.png", 
       width = 15, height = 15, units ='cm', dpi = 300)
