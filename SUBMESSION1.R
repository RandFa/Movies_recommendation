#loading required libraries
library(tidyverse)
library(caret)
library(dplyr)
library(lubridate)
library(data.table)
#downloading, wrangling and partitioning data folowing instructin provided in the course
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
# saving edx and validation sets
save(edx, file = "rda/edx.rda")
save(validation, file = "rda/validation.rda")


#MY ANALYSIS


#loading edx set
load("rda/edx.rda")


#exploratory data analysis
#examining different genres combination and their average ratings
#top rated genres
edx %>% group_by(genres) %>%summarize(count= n(), avg = mean(rating))%>%
  arrange(desc(avg)) %>% head()

#least rated genres
edx %>% group_by(genres) %>% summarize(count= n(), avg = mean(rating))%>%
   arrange(desc(avg)) %>% tail()

#adding release year column, rating year, difference between release and rating year
edx <-edx%>% mutate(relyear=as.numeric(str_match(title, "\\((\\d{4})\\)$")[,2]),
                    ratyear = year(as.Date(as.POSIXct(timestamp, origin="1970-01-01"))),
                    yeardif = ratyear-relyear)

#plotting average rating against release year for years with more than 1000 rating
edx %>% group_by(relyear) %>%  filter(n()>1000) %>% summarize(avg = mean(rating))%>% ggplot(aes(relyear, avg)) + geom_point()
#adding column to indicate if release year < 1978 or later
edx <-edx %>%  mutate(ygroup = ifelse(relyear<1978, 1, 0))
#calculating the mean for each group to approve the conclusion
edx %>% group_by(ygroup) %>% summarize(avg = mean(rating))

#plotting average rating against difference between release and rating year filtering for more than 1000 rating
edx %>% group_by(yeardif) %>% filter(n()>1000) %>% summarize(avg = mean(rating)) %>% ggplot(aes(yeardif, avg)) + geom_point()

# selecting chosen columns
edx <- edx %>% select(movieId, userId, ygroup, yeardif, genres, rating)



#dividing edx into training (80%) and testing (20) sets 
#after setting the seed to make results reproducable
set.seed(1999, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

#making sure that all userId and movieId in the test set are present in train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# generating different lambda values to test and find optimum
lambdas <- seq(0, 10, 1)

#building regression model and testing on lambdas
rmses <- sapply(lambdas, function(l){
  #calualting the rating mean of the training dataset (part of edx)
  mu <- mean(train_set$rating)
  # introducing movieId bias 
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # introducing userId bias 
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # introducing genres bias
  b_g <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i -b_u - mu)/(n()+l))
  #introducing year difference bias
  b_yd <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(yeardif) %>%
    summarize(b_yd = sum(rating - b_i -b_u-b_g - mu)/(n()+l))
  #introducing release year group bias
  b_yg <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_yd, by="yeardif") %>%
    group_by(ygroup) %>%
    summarize(b_yg = sum(rating - b_i -b_u-b_g -b_yd- mu)/(n()+l))
  # testing the model on the test set (part of edx)
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_yd, by = "yeardif")%>%
    left_join(b_yg, by="ygroup") %>%
    mutate(pred = mu + b_i + b_u+b_g+b_yd+b_yg) %>%
    pull(pred)
  #calculating rmse
  return(RMSE(predicted_ratings, test_set$rating))
})

#plotting lambdas against rmses
plot(lambdas, rmses)

#finding the minimum value of rmses
min(rmses)

#finding optimum lambda that produces the previous minimum rmse
bestl = lambdas[which.min(rmses)]

#calualting the rating mean of the entire edx dataset
mu <- mean(edx$rating)

# training the final model on the entire edx dataset using the optimum lambda

# introducing moviId bias first to the mean of the whole edx set
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+bestl))

# introducing userId bias 
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+bestl))

#introducing genres bias 
b_g <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i -b_u - mu)/(n()+bestl))

#introducing year difference bias
b_yd <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(yeardif) %>%
  summarize(b_yd = sum(rating - b_i -b_u-b_g - mu)/(n()+bestl))

#introducing release year group bias
b_yg <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_yd, by="yeardif") %>%
  group_by(ygroup) %>%
  summarize(b_yg = sum(rating - b_i -b_u-b_g -b_yd- mu)/(n()+bestl))

#loading validation set
load("rda/validation.rda")

#adding the required columns to the validation set and selecting the ones included in the model
validation <-validation%>% mutate(relyear=as.numeric(str_match(title, "\\((\\d{4})\\)$")[,2]),
                                  ratyear = year(as.Date(as.POSIXct(timestamp, origin="1970-01-01"))),
                                  yeardif = ratyear-relyear)

validation <-validation %>%  mutate(ygroup = ifelse(relyear<1978, 1, 0))
validation <- validation %>% select(movieId, userId, ygroup, yeardif, genres, rating)

# Testing the final model on the validation set.
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_yd, by = "yeardif")%>%
  left_join(b_yg, by="ygroup") %>%
  mutate(pred = mu + b_i + b_u+b_g+b_yd+b_yg) %>%
  pull(pred)
#Calculating the RMSE of applying the final model on the validation set.
RMSE(predicted_ratings, validation$rating)
