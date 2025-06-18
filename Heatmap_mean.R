
#Read CSV
data <- read.csv("C:/Users/raiya/OneDrive/Documents/Thesis_data/Hexagon_Mean_SD_predictors.csv")


# Variable names 
soil_vars <- c("Mean_Bulk_Dens") 
climate_vars <- c("Mean_Temp", "Mean_P") 
biomass_var <- "Mean_Biomass" 

# Selected variables for correlation analysis 
selected_data <- data[, c(soil_vars, climate_vars, biomass_var)]

# Calculate correlation matrix
corr_matrix <- cor(selected_data, method = "pearson")

# Melt the correlation matrix for ggplot2
melted_corr <- melt(corr_matrix)

# Create the correlation heatmap 
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name = "Correlation") +
  theme(legend.title = element_text(size = 14, color = "black")) +
  
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45,size = 14, color = "black", vjust = 1, hjust = 1),
        axis.text.y = element_text(angle = 45,size = 14, color = "black", vjust = 1, hjust = 1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) +
  coord_fixed() +
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 6) 
 

# Save the plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/correlation heatmap mean.png", 
       width = 6, height = 6, dpi = 300, scale = 1.2)

# Scatter plot to highlight collinearity between Mean_Bulk_Dens and Mean_Temp
ggplot(selected_data, aes(x = Mean_Bulk_Dens, y = Mean_Temp)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(x = "Mean Soil Bulk Density (tm⁻³)",
       y = "Mean Temperature (℃)") +
  theme_minimal(base_size = 14) +
  theme (axis.title.x = element_text(size = 14, color = "black"),
         axis.title.y = element_text(size = 14, color = "black"),
         axis.text.x = element_text(size = 14, color = "black" ),
         axis.text.y = element_text(size = 14, color = "black"))

# Save the plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/soilbulkdens temp cor mean.png", 
       width = 15, height = 15, units = 'cm', dpi = 300)

# Scatter plot to highlight collinearity between Mean_Bulk_Dens and Mean_Prec
ggplot(selected_data, aes(x = Mean_Bulk_Dens, y = Mean_P)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(x = "Mean Soil Bulk Density (tm⁻³)",
       y = "Mean Precipitation (mmday⁻¹)")  +
  theme_minimal(base_size = 14) +
  theme (axis.title.x = element_text(size = 14, color = "black"),
         axis.title.y = element_text(size = 14, color = "black"),
         axis.text.x = element_text(size = 14, color = "black" ),
         axis.text.y = element_text(size = 14, color = "black"))

# Save the plot
ggsave("C:/Users/raiya/OneDrive/Documents/Thesis_data/soilbulkdens prec cor mean.png", 
       width = 15, height = 15, units = 'cm', dpi = 300)



