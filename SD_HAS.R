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
input_csv <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/HAS_data.csv"
data_HAS <- read.csv(input_csv)

# Clean and preprocess the data
data_cleaned_HAS <- data_HAS %>%
  select(-c(Mean_Biomass, GRID_ID)) %>%      # Remove Mean_Biomass and GRID_ID columns
  na.omit()    #Remove N/A values

# Separate features and target
target_var_HAS <- "SD_Biomass"
predictors_HAS <- setdiff(names(data_cleaned_HAS), target_var_HAS)

# Prepare feature matrix (x) and target (y)
x_sd_HAS <- data_cleaned_HAS[, predictors_HAS]
y_sd_HAS <- data_cleaned_HAS[[target_var_HAS]]

# Train-test split (80% train, 20% test)
set.seed(52)
train_indices_HAS <- createDataPartition(y_sd_HAS, p = 0.8, list = FALSE)
x_train_sd_HAS <- x_sd_HAS[train_indices_HAS, ]
y_train_sd_HAS <- y_sd_HAS[train_indices_HAS]
x_test_sd_HAS <- x_sd_HAS[-train_indices_HAS, ]
y_test_sd_HAS <- y_sd_HAS[-train_indices_HAS]

# Train Random Forest model
set.seed(52)
rf_model_sd_HAS <- randomForest(
  x = x_train_sd_HAS,
  y = y_train_sd_HAS,
  importance = TRUE,
  ntree = 500
)

# Print model summary
print(rf_model_sd_HAS)

# Extract and normalize variable importance
var_imp_HAS <- data.frame(
  variable = rownames(rf_model_sd_HAS$importance),
  Overall = rf_model_sd_HAS$importance[, "IncNodePurity"]
) %>%
  mutate(ScaledImportance = 100 * Overall / max(Overall)) %>%
  arrange(desc(ScaledImportance))

# Plot variable importance
ggplot(var_imp_HAS, aes(x = reorder(variable, ScaledImportance), y = ScaledImportance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6, color = "black", linewidth = 0.3) +
  labs(x = "Variables", y = "Variable Importance") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 12, angle = 45, hjust = 1, color = "black"),
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14, color = "black")
  )

# Predictions
y_pred_train_sd_HAS <- predict(rf_model_sd_HAS, x_train_sd_HAS)
y_pred_sd_HAS <- predict(rf_model_sd_HAS, x_test_sd_HAS)

# Metrics
r2_train_sd_HAS <- caret::R2(y_pred_train_sd_HAS, y_train_sd_HAS)
rmse_train_sd_HAS <- Metrics::rmse(y_train_sd_HAS, y_pred_train_sd_HAS)
mae_train_sd_HAS <- Metrics::mae(y_train_sd_HAS, y_pred_train_sd_HAS)

r2_test_sd_HAS <- caret::R2(y_pred_sd_HAS, y_test_sd_HAS)
rmse_test_sd_HAS <- Metrics::rmse(y_test_sd_HAS, y_pred_sd_HAS)
mae_test_sd_HAS <- Metrics::mae(y_test_sd_HAS, y_pred_sd_HAS)

cat(sprintf(
  "\nSD biomass HAS - training performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_train_sd_HAS, rmse_train_sd_HAS, mae_train_sd_HAS
))
cat(sprintf(
  "SD biomass HAS - test performance:\n R-squared = %.3f\n RMSE = %.3f\n MAE = %.3f\n",
  r2_test_sd_HAS, rmse_test_sd_HAS, mae_test_sd_HAS
))

# Validation Plot (Fixed Axis)
validation_data_sd_HAS <- data.frame(
  Actual = y_test_sd_HAS,
  Predicted = y_pred_sd_HAS
)

# Fixed axis for consistency across districts
axis_limits <- c(0, 400)
breaks <- seq(0, 400, 100)

ggplot(validation_data_sd_HAS, aes(x = Actual, y = Predicted)) +
  geom_point(color = "black", alpha = 0.5, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", size = 0.5) +
  coord_fixed(ratio = 1) +
  scale_x_continuous(limits = axis_limits, breaks = breaks) +
  scale_y_continuous(limits = axis_limits, breaks = breaks) +
  labs(
    x = "Actual SD Biomass HAS (g/m²)",
    y = "Predicted SD Biomass HAS (g/m²)"
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
           x = axis_limits[2] - 10,
           y = axis_limits[1] + 50,
           label = sprintf("R²    %.2f\nRMSE  %.2f\nMAE   %.2f", 
                           r2_test_sd_HAS, rmse_test_sd_HAS, mae_test_sd_HAS),
           hjust = 1, size = 5, color = "black")
# Save the validation plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/val_SD_HAS.png", 
       width = 15, height = 15, units ='cm', dpi = 300)