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

# Load data
data <- read.csv("C:/Users/raiya/OneDrive/Documents/Thesis_data/Hexagon_Mean_SD_predictors.csv")

# Define predictor variables
predictor_vars <- c("Mean_SWF","Mean_SHDI",	"SD_SHDI",	"Mean_Elev",	"Mean_Slope","Mean_Aspect","Mean_Temp","Mean_Rs",
                    "Mean_P","Mean_Soil_pote","SD_Rs","SD_P","SD_Temp", "SD_Soil_pote", "SD_Elev",	"SD_Aspect",	"SD_Slope",	
                    "Mean_Bulk_Dens",	"SD_Bulk_Dens",	"Mean_Ca_Exg",	"SD_Ca_Exg",	"Mean_Clay",	"SD_Clay",	"Mean_Sand",	"SD_Sand",	"Mean_Silt", "SD_Silt")

# Check if all predictor variables exist
missing_vars <- setdiff(predictor_vars, names(data))
if (length(missing_vars) > 0) {
  stop(paste("Error: Missing variables in dataset:", paste(missing_vars, collapse = ", ")))
}

# Select predictor variables and target variable
X <- data[, predictor_vars]
y <- data$SD_Biomass  # Target variable
y <- as.numeric(y)

# Train the Random Forest model
rf_model <- randomForest(X, y, importance = TRUE, ntree = 500)
print(rf_model)

# Define the SHAP prediction wrapper function
predict_function <- function(object, newdata) {
  predict(object, newdata = newdata)
}

# Compute SHAP values
shap_values <- fastshap::explain(rf_model, X = X, nsim = 50, pred_wrapper = predict_function)

# Convert SHAP values to a dataframe
shap_df <- as.data.frame(shap_values)

# Apply log transformation first
shap_df_log <- shap_df %>%
  mutate(across(everything(), ~ log1p(abs(.))))  # log1p ensures no log(0) issues

# Now apply Min-Max Scaling
shap_df_normalized <- shap_df_log %>%
  mutate(across(everything(), ~ (. - min(.)) / (max(.) - min(.))))


summary(shap_df_normalized)



# Add GRID_ID column if it exists in the dataset
if ("GRID_ID" %in% colnames(data)) {
  shap_df_normalized$GRID_ID <- data$GRID_ID
  shap_df_normalized <- shap_df_normalized %>% select(GRID_ID, everything())
}

# Save SHAP values as a CSV
output_file <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/SHAP_Values_Normalized_WW_SD.csv"
write.csv(shap_df_normalized, output_file, row.names = FALSE)


print(paste("SHAP values saved as:", output_file))

# Compute mean absolute SHAP values for each feature
shap_feature_importance <- shap_df_normalized %>%
  select(-GRID_ID) %>%  # Remove GRID_ID if it exists
  summarise(across(everything(), ~ mean(abs(.), na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "Feature", values_to = "SHAP_Value") %>%
  arrange(desc(SHAP_Value))  # Sort in descending order

# Create a vertical bar plot with renamed features
ggplot(shap_feature_importance, aes(x = reorder(Feature, -SHAP_Value), y = SHAP_Value)) +
  geom_bar(stat = "identity", fill = "khaki", color = "black") +  
  labs(
       x = "Feature",
       y = "Mean Absolute SHAP Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))