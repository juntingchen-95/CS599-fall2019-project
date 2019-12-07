# Import the library ggplot2
library(ggplot2)

# Set the working directory the project folder
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)

# Read data from the csv file
result_data = read.csv(paste(dir, 'result.csv', sep='/'), header = TRUE)

plot_1 <- ggplot() + 
  # Set x axis as log10
  scale_x_log10() + 
  # Plot lines
  geom_line(data = result_data, mapping = aes(x = number, y = max, color = loss_type), size = 1.5, lineend = "round") + 
  xlab('Number of Output') + 
  ylab('Max moves') 

# Save the plot as PDF format
ggsave(filename = paste(dir, 'result_figure_1.pdf', sep='/'), plot = plot_1, width = 10, height = 8)

plot_2 <- ggplot() + 
  scale_x_log10() + 
  geom_line(data = result_data, mapping = aes(x = number, y = average, color = loss_type), size = 1.5, lineend = "round") + 
  xlab('Number of Output') + 
  ylab('Average moves') 

ggsave(filename = paste(dir, 'result_figure_2.pdf', sep='/'), plot = plot_2, width = 10, height = 8)
