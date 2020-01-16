library(tidyverse)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(randomForest)
library(doSNOW)

# ----------------- Part 1: Intro + Objective -----------------
# load data
data.raw = read.csv("../Data/spam.csv", stringsAsFactors = F, fileEncoding="latin1")

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
# conclusion: text length may help with determining spam/ham


# ----------------- Part 2: Train/Test Split -----------------
# train/test split (70%/30% stratified split)
set.seed(32984)
indicies = createDataPartition(data.raw$Label, times = 1, p = 0.7, list = F) 
train.data = data.raw[indicies,]
test.data = data.raw[-indicies,]

# verifying that'createDataPartition' preserved the original proportions of labels
prop.table(table(train.data$Label))
prop.table(table(test.data$Label))

# About BOW or DFM:
# BOW(bag-of-words) or DFM(docuemnt frequency matrix) is a common model to convert
# text into dataframes. Tokenize words into columns and let documents correspond
# to rows, where the cells' values represent the frequency of the token in the document
# Considerations: casing, punctuation, numbers, every word, symbols, similar words
# Thus, pre-processing is crucial for text analytics!

# ----------------- Part 3: Preproessing Pipeline + DFM -----------------
# exploring some issues to be handled with pre-processing
train.data$Text[21]
train.data$Text[38]
train.data$Text[357]

# tokenize SMS text messages
train.tokens = tokens(train.data$Text, what = "word",
                      remove_numbers = T, remove_punct = T,
                      remove_symbols = T, remove_hyphens = T) 

# see examples of the result
train.tokens[[300]]
train.tokens[[357]]

# lower case all tokens
train.tokens = tokens_tolower(train.tokens)
train.tokens[[357]]

# use quanteda's built-in stopword list for English
# NOTE: always inspect stopword lists for applicability to the problem/domain
train.tokens = tokens_select(train.tokens, stopwords(), selection = "remove")
# see examples of the result
train.tokens[[357]]

# perform stemming on the tokens
train.tokens = tokens_wordstem(train.tokens, language = "english")
train.tokens[[357]]

# create a BOW model
train.tokens.dfm = dfm(train.tokens, tolower = F)
train.tokens.df = as.data.frame(train.tokens.dfm)
dim(train.tokens.df)
view(train.tokens.df[1:10, 1:100]) # problem: sparsity & curse of dimensionality
colnames(train.tokens.df[1:30])

# ----------------- Part 4: BOW Model -----------------
# Per best practice, we will leverage cross validation as the basis of the modeling process

# set up a feature data frame with labels
train.tokens.data = cbind(Label = train.data$Label,
                          train.tokens.df)
# colnames need some preprocessing
names(train.tokens.data[c(146, 148, 235, 238)])
# cleanup column names in an automatic fashion
names(train.tokens.data) = make.names(names(train.tokens.data))

# use caret to create stratified folds for 10-fold CV repeated 3 times
set.seed(48743)
cv.folds = createMultiFolds(train.data$Label, k = 10, times = 3)
cv.cntrl = trainControl(method = "repeatedcv", number = 10,
                        repeats = 3, index = cv.folds)

# our data frame is non-trivial in size, so we can use multi-core training in parallel
# use the doSNOW package to allow for multi-core processing
# on mac use `$ sysctl hw.physicalcpu hw.logicalcpu` to check the number of available cores

# time the process
start.time = Sys.time()

# number of logical cores
cl = makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

# train the model
rpart.cv.1 = train(Label ~ ., data = train.tokens.data,
                   method = "rpart", trControl = cv.cntrl,
                   tuneLength = 50)
# stop cluster
stopCluster(cl)

# calculate the time of execution
total.time = Sys.time() - start.time
total.time

# see results
rpart.cv.1

# ----------------- Part 5: TF-IDF -----------------
# DFM's potential problems:
# 1) Longer documents will tend to have higher term counts
# 2) Terms that appear frequently across the corpus aren't as important
# We can improve by 
# 1) normalize documents based on their length 
# 2) penalize terms with high frequency across corpus
# Which leads to the method of TF-IDF

# freq(t,d) = the count of the instances of the term t in document d
# TF(t,d) = the proportion of the count of term t in document d = freq(t,d)/sum[freq(ti,d)]
# count(t) = the count of documents in the corpus in which the term t is present
# IDF(t) = log(N/count(t)), where N is the total number of distinct documents (Inverse Document Frequency)
# Finally, TF-IDF(t,d) = TF(t,d)*IDF(t)




# ----------------- Output HTML -----------------
rmarkdown::render("part1.R")