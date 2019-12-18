library(tidyverse)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(randomForest)

# ----------------- Part 1 -----------------
# load data
data.raw = read.csv("../Data/spam.csv", stringsAsFactors = F)

# data preprocessing
data.raw = data.raw[1:2] # keep only the useful columns
colnames(data.raw) = c("Label", "Text") # rename columns
data.raw$Label = as.factor(data.raw$Label)

# check if there are missing values
length(which(!complete.cases(data.raw)))

# distribution of label
prop.table(table(data.raw$Label))

# count the length of each text
data.raw$TextLength = nchar(data.raw$Text)
summary(data.raw$TextLength)

# visualize text length distribution
ggplot(data.raw, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(bins = 80) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths")

# summary statistics
data.raw %>%
  group_by(Label) %>%
  summarise("mean" = mean(TextLength),
            "median" = median(TextLength),
            "sd" = sd(TextLength))
# t-test
with(data.raw, shapiro.test(TextLength[Label == "ham"]))
with(data.raw, shapiro.test(TextLength[Label == "spam"]))

res.ftest <- var.test(TextLength ~ Label, data = data.raw)
res.ftest

res <- t.test(data.raw$TextLength[data.raw$Label=="ham"], 
              data.raw$TextLength[data.raw$Label=="spam"], var.equal = FALSE)


# ----------------- Part 2 -----------------
