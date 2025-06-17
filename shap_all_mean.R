
# Install required packages if not installed
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("DescTools", quietly = TRUE)) install.packages("DescTools")
if (!requireNamespace("fastshap", quietly = TRUE)) install.packages("fastshap")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr")

# Load necessary libraries
library(randomForest)
library(fastshap)
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)
library(DescTools)

# Load dataset
input_file <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/Hexagon_Mean_SD_predictors.csv"
data <- read.csv(input_file)

# Define predictor variables
predictor_vars <- c("Mean_SWF", "Mean_SHDI", "SD_SHDI", "Mean_Elev", "Mean_Slope", "Mean_Aspect",
                    "Mean_Temp", "Mean_Rs", "Mean_P", "Mean_Soil_pote", "SD_Rs", "SD_P", "SD_Temp",
                    "SD_Soil_pote", "SD_Elev", "SD_Aspect", "SD_Slope", "Mean_Bulk_Dens", "SD_Bulk_Dens",
                    "Mean_Ca_Exg", "SD_Ca_Exg", "Mean_Clay", "SD_Clay", "Mean_Sand", "SD_Sand", "Mean_Silt", "SD_Silt")

# Check if all predictor variables are present
missing_vars <- setdiff(predictor_vars, names(data))
if (length(missing_vars) > 0) {
  stop(paste("Error: Missing variables in dataset:", paste(missing_vars, collapse = ", ")))
}

# Define features (X) and new target variable (y)
X <- data[, predictor_vars]
y <- data$Mean_Biomass
y <- as.numeric(y)

# Train Random Forest model for Mean_Biomass
set.seed(52)
rf_model <- randomForest(X, y, importance = TRUE, ntree = 500)
print(rf_model )

# Define prediction wrapper function for SHAP
predict_function <- function(object, newdata) {
  predict(object, newdata = newdata)
}

# Compute SHAP values
shap_values <- fastshap::explain(rf_model , X = X, nsim = 50, pred_wrapper = predict_function)

# Convert to data frame
shap_df <- as.data.frame(shap_values)

# Apply log transformation and min-max normalization
shap_df_log <- shap_df %>%
  mutate(across(everything(), ~ log1p(abs(.))))  # Avoid log(0)

shap_df_normalized <- shap_df_log %>%
  mutate(across(everything(), ~ (. - min(.)) / (max(.) - min(.))))

# Add GRID_ID if available
if ("GRID_ID" %in% colnames(data)) {
  shap_df_normalized$GRID_ID <- data$GRID_ID
  shap_df_normalized <- shap_df_normalized %>% select(GRID_ID, everything())
}

# Save normalized SHAP values
output_file <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/SHAP__Normalized__Mean.csv"
write.csv(shap_df_normalized, output_file, row.names = FALSE)
print(paste("SHAP values saved to:", output_file))

# Compute mean absolute SHAP value for each feature
shap_feature_importance <- shap_df_normalized %>%
  select(-GRID_ID) %>%  # Remove GRID_ID if present
  summarise(across(everything(), ~ mean(abs(.), na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") %>%
  arrange(desc(mean_abs_shap))

# Create SHAP bar plot
ggplot(shap_feature_importance, aes(x = reorder(feature, -mean_abs_shap), y = mean_abs_shap)) +
  geom_bar(stat = "identity", fill = "khaki", color = "black") +
  labs(
    x = "Feature",
    y = "Mean Absolute SHAP Value",
    title = "Feature Importance (SHAP) for Mean Biomass"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
