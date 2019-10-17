# Import the library ggplot2
library(ggplot2)

# Set the working directory the project folder
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)

# Read the training data from the csv file
training_data <- read.csv(paste(dir, "dataset/training_set.csv", sep='/'), header = TRUE)

# Read the file which produced by the KNN program (main.py)
result <- read.csv(paste(dir, "result.csv", sep='/'), header = TRUE)

# Read the bayes probability data from the csv file
bayes_prob <- read.csv(paste(dir, "dataset/bayes_prob.csv", sep='/'), header = TRUE)

ggplot() + 
  # Set colors of points
  scale_color_manual(values = c("#5EB6E9", "#E99F00")) +
  # Plot the background grid points
  geom_point(data = result, mapping = aes(x = x, y = y, color = factor(type)), shape = 15, size = 0.3) +
  # Plot the training data (circles)
  geom_point(data = training_data, mapping = aes(x = x, y = y, color = factor(type)), shape = 1, size = 3, stroke = 1.7) +
  # Plot the KNN contour (black lines)
  geom_contour(data = result, mapping = aes(x = x, y = y, z = type), 
               color = "black", bins = 2, breaks = c(0.5), size = 0.8, lineend = "round") +
  # Plot the bayes decision boundary (purple dashed lines)
  geom_contour(data = bayes_prob, mapping = aes(x = x, y = y, z = prob), 
               color = "purple", bins = 2, breaks = c(0.5), size = 1, linetype = 2, lineend = "round") +
  # Set the plot style
  theme(panel.background = element_rect(color = "black", fill = "white"), panel.grid = element_blank(),
        axis.title = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), legend.position = "none")

# Save the plot as PDF format
ggsave(filename = paste(dir, "result_figure.pdf", sep='/'), width = 10, height = 9)