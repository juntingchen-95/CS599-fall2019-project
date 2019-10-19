# Import the library ggplot2
library(ggplot2)

# Set the working directory the project folder
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)

# Read the training data from the csv file
training_data <- read.csv(paste(dir, "dataset/training_set.csv", sep='/'), header = TRUE)

# Read the test result file which produced by the KNN program (main.py)
test_result <- read.csv(paste(dir, "test_set_result.csv", sep='/'), header = TRUE)

# Read the training test result file
training_test_result <- cbind(read.csv(paste(dir, "training_set_result.csv", sep='/'), header = TRUE), training_data$type)

# Read the bayes probability data from the csv file
bayes_prob <- read.csv(paste(dir, "dataset/bayes_prob.csv", sep='/'), header = TRUE)

# Read the marginal probability data from the csv file
marginal <- read.csv(paste(dir, "dataset/marginal.csv", sep='/'), header = TRUE)

# Calculate training error
training.error <- sum(1 * I(training_test_result$`training_data$type` != training_test_result$type)) / nrow(training_test_result)

# Calculate test error
logit <- glm(type~cbind(x,y),family=binomial(link='logit'),data=training_data)
pred <- predict(logit, newdata = test_result)
test.error <- sum(marginal$marginal * (bayes_prob$prob * I(pred < 0) + (1 - bayes_prob$prob) * I(pred >= 0)))

# Calcualte bayes error
bayes.error <- sum(marginal$marginal * (bayes_prob$prob * I(bayes_prob$prob < 0.5) + 
                                          (1 - bayes_prob$prob) * I(bayes_prob$prob >= 0.5)))

plot <- ggplot() + 
  # Set colors of points
  scale_color_manual(values = c("#5EB6E9", "#E99F00")) +
  # Plot the background grid points
  geom_point(data = test_result, mapping = aes(x = x, y = y, color = factor(type)), shape = 15, size = 0.3) +
  # Plot the training data (circles)
  geom_point(data = training_data, mapping = aes(x = x, y = y, color = factor(type)), shape = 1, size = 3, stroke = 1.7) +
  # Plot the KNN contour (black lines)
  geom_contour(data = test_result, mapping = aes(x = x, y = y, z = type), 
               color = "black", bins = 2, breaks = c(0.5), size = 0.8, lineend = "round") +
  # Plot the bayes decision boundary (purple dashed lines)
  geom_contour(data = bayes_prob, mapping = aes(x = x, y = y, z = prob), 
               color = "purple", bins = 2, breaks = c(0.5), size = 1, linetype = 2, lineend = "round") +
  # Set the plot style
  theme(panel.background = element_rect(color = "black", fill = "white"), panel.grid = element_blank(),
        axis.title = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), legend.position = "none") +
  # Annotate error values on the figure
  annotate("rect", xmin = -2.61, xmax = -0.95, ymin = -2.02, ymax = -1.45, fill = "white", color = "white") +
  annotate("text", label = paste("Training Error:\nTest Error:\nBayes Error:"), x = -2.6, y = -1.8, size = 5, hjust = 0) +
  annotate("text", label = paste(round(training.error, 3), round(test.error, 3), round(bayes.error, 3), sep='\n'), 
           x = -1.45, y = -1.8, size = 5, hjust = 0)

# Save the plot as PDF format
ggsave(filename = paste(dir, "result_figure.pdf", sep='/'), plot = plot, width = 10, height = 9)
