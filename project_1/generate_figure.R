library(ggplot2)
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
training_data <- read.csv(paste(dir, "dataset/training_set.csv", sep='/'), header = TRUE)
result <- read.csv(paste(dir, "result.csv", sep='/'), header = TRUE)
ggplot() + 
  geom_point(data = result, mapping = aes(x = x, y = y, color = factor(type)), shape = 15, size = 0.3) +
  scale_color_manual(values = c("#5EB6E9", "#E99F00")) +
  geom_contour(data = result, mapping = aes(x = x, y = y, z = type), color = "black", bins = 1) +
  geom_point(data = training_data, mapping = aes(x = x, y = y, color = factor(type)), shape = 1, size = 3, stroke = 1.7)
ggsave(filename = paste(dir, "result.pdf", sep='/'), width = 10, height = 8)