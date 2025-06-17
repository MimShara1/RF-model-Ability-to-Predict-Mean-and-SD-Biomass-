library(pdp)
library(randomForest)
library(dplyr)
library(ggplot2)
library(caret)

# Set base path and working directory 
base_path <- "C:/Users/raiya/OneDrive/Documents/Thesis_data/"
setwd(base_path)

# Load dataset 
data_reg <- read.csv(paste0(base_path, "Hexagon_Mean_SD_predictors.csv"))

# Clean and preprocess the data 
data_cleaned_reg <- data_reg %>%
  select(-c(Mean_Biomass, GRID_ID)) %>%  # Remove Mean_Biomass and GRID_ID columns
  na.omit()                             # Remove rows with NA values

# Separate features and target 
target_var_reg <- "SD_Biomass"
predictors_reg <- setdiff(names(data_cleaned_reg), target_var_reg)

# Prepare feature matrix (x) and target (y)
x_sd_reg <- data_cleaned_reg[, predictors_reg]
y_sd_reg <- data_cleaned_reg[[target_var_reg]]

# Train-test split (80/20) 
set.seed(52)
train_indices_sd_reg <- createDataPartition(y_sd_reg, p = 0.8, list = FALSE)

x_train_sd_reg <- x_sd_reg[train_indices_sd_reg, , drop = FALSE]
x_test_sd_reg <- x_sd_reg[-train_indices_sd_reg, , drop = FALSE]

y_train_sd_reg <- y_sd_reg[train_indices_sd_reg]
y_test_sd_reg <- y_sd_reg[-train_indices_sd_reg]

# Train the Random Forest model 
set.seed(52)
rf_model_sd <- randomForest(
  x = x_train_sd_reg,
  y = y_train_sd_reg,
  importance = TRUE,
  ntree = 500
)

print(rf_model_sd)

# Compute PDPs for predictors 
predictors_sd_reg <- colnames(x_train_sd_reg)
pdp_list_sd_reg <- list()

for (pred in predictors_sd_reg) {
  print(paste("Computing PDP for", pred))
  pdp_res <- tryCatch({
    partial(
      object = rf_model_sd,
      pred.var = pred,
      train = x_train_sd_reg,
      grid.resolution = 100
    )
  }, error = function(e) {
    print(paste("Error PDP:", pred, e$message))
    return(NULL)
  })
  if (is.null(pdp_res)) next
  colnames(pdp_res) <- c("Predictor_Value", "PDP_Value")
  pdp_res$Predictor <- pred
  pdp_list_sd_reg[[pred]] <- pdp_res
}

pdp_combined_sd_reg <- bind_rows(pdp_list_sd_reg)

# Normalize PDP values per predictor (min-max scaling) 
pdp_combined_sd_reg <- pdp_combined_sd_reg %>%
  group_by(Predictor) %>%
  mutate(
    PDP_Value_Normalized = (PDP_Value - min(PDP_Value, na.rm = TRUE)) /
      (max(PDP_Value, na.rm = TRUE) - min(PDP_Value, na.rm = TRUE))
  ) %>%
  ungroup() %>%
  select(Predictor, Predictor_Value, PDP_Value_Normalized) %>%
  rename(PDP_Value = PDP_Value_Normalized)

# Save PDP CSV 
write.csv(pdp_combined_sd_reg, "SD_PDP_Values_Combined.csv", row.names = FALSE)
print(" SD Biomass PDP values saved as SD_PDP_Values_Combined.csv")

# Create output folder for SD Biomass PDP plots 
output_dir_sd_reg <- paste0(base_path, "PDP_Plots_SD_Biomass_Normalized/")
dir.create(output_dir_sd_reg, showWarnings = FALSE)

#  Feature units mapping 
feature_units <- list(
  "Mean_Temp" = "℃",
  "SD_Temp" = "℃",
  "Mean_P" = "mm day⁻¹",
  "SD_P" = "mm day⁻¹",
  "Mean_Rs" = "mJ m⁻² day⁻¹",
  "SD_Rs" = "mJ m⁻² day⁻¹",
  "Mean_Elev" = "m.a.s.l",
  "SD_Elev" = "m.a.s.l",
  "Mean_Slope" = "º",
  "SD_Slope" = "º",
  "Mean_Aspect" = "º",
  "SD_Aspect" = "º",
  "Mean_Silt" = "%",
  "SD_Silt" = "%",
  "Mean_Clay" = "%",
  "SD_Clay" = "%",
  "Mean_Sand" = "%",
  "SD_Sand" = "%",
  "Mean_Bulk_Dens" = "tm⁻³",
  "SD_Bulk_Dens" = "tm⁻³",
  "Mean_Ca_Exg" = "cmol(+)kg⁻¹",
  "SD_Ca_Exg" = "cmol(+)kg⁻¹",
  "Mean_Soil_pote" = "%",
  "SD_Soil_pote" = "%",
  "Mean_SWF" = "%",
  "Mean_SHDI" = "",
  "SD_SHDI" = ""
)

#  Plot PDPs with normalized values 
for (pred in unique(pdp_combined_sd_reg$Predictor)) {
  print(paste("Plotting PDP for", pred))
  
  pdp_df <- pdp_combined_sd_reg %>%
    filter(Predictor == pred) %>%
    select(Predictor_Value, PDP_Value)
  
  unit <- feature_units[[pred]]
  x_label <- ifelse(!is.null(unit) && unit != "", paste0(pred, " (", unit, ")"), pred)
  
  p <- ggplot(pdp_df, aes(x = Predictor_Value, y = PDP_Value)) +
    geom_line(color = "black", linewidth = 0.3) +
    theme_classic(base_size = 12) +
    labs(x = x_label, y = "SD Biomass") +
    theme(
      axis.title = element_text(size = 12, color = "black"),
      axis.text = element_text(size = 12, color = "black"),
      aspect.ratio = 1
    ) +
    scale_x_continuous() +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1), minor_breaks = seq(0, 1, 0.05))
  
  ggsave(filename = paste0(output_dir_sd_reg, pred, "_PDP_Normalized.png"),
         plot = p, width = 7.56, height = 5.67, units = "cm", dpi = 300)
}
