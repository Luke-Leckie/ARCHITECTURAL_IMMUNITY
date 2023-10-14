rm(list=ls())
gc()

library(ggplot2)
library(HelpersMG)
library(glmm)
library(lme4)
library(lmerTest)
library(MASS)
library(vegan)
library(multcomp)
library(segmented)
library(splines)
library(lspline)
library(caret)
library(dplyr)
library(tidyr)
library(ggbreak)
library(report)
library(rsq)
library(car)
library(emmeans)
library(segmented)
library(lme4)
qq_plot <- function(model, main) {
  qqnorm(resid(model), main = main)
  qqline(resid(model))
}
cohens_d <- function(data, variable_col, group_col, group1, group2) {
  # Extract the variable and group columns
  variable <- data[[variable_col]]
  group <- data[[group_col]]
  
  # Calculate Means and Standard Deviations for each group
  mean_group1 <- mean(variable[group == group1], na.rm = TRUE)
  sd_group1 <- sd(variable[group == group1], na.rm = TRUE)
  
  mean_group2 <- mean(variable[group == group2], na.rm = TRUE)
  sd_group2 <- sd(variable[group == group2], na.rm = TRUE)
  
  # Calculate the number of observations in each group
  n1 <- sum(group == group1, na.rm = TRUE)
  n2 <- sum(group == group2, na.rm = TRUE)
  
  # Calculate Pooled Standard Deviation
  sd_pooled <- sqrt((sd_group1^2 + sd_group2^2) / 2)
  # Calculate Cohen’s d
  cohens_d <- (mean_group1 - mean_group2) / sd_pooled
  
  return(cohens_d)
}

DF <- read.csv('/home/ll16598/Documents/ARCHITECTURAL_IMMUNITY/ABM_CODE/1_combined_social_analysis.csv', header=TRUE)

DF$day_list


##DENSITY
DF$variable<-DF$density_list
hist(DF$variable)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt2 <- lmer(sqrt(variable) ~ treat_list+day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_gamma <- glmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), family='Gamma', data = DF)
C <- max(DF$variable) + 1
DF$reflected_variable <- C - DF$variable
hist(DF$reflected_variable)
model_gamma <- glmer(reflected_variable ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)
model_gamma2 <- glmer(reflected_variable ~ treat_list+day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)


anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model_gamma2)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma2
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=2)


mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0),
  "PathogenMinusShamMON"=c(0,-1,0))
differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')

cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
cd_wed
cd_mon


###Efficiency
DF$variable<-DF$efficiency_list
hist(DF$variable)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|iteration)+(1|t_chunks), data = DF)
model_sqrt2 <- lmer(sqrt(variable) ~ treat_list+day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
C <- max(DF$variable) + 1
DF$reflected_variable <- C - DF$variable

hist(DF$reflected_variable)
model_gamma <- glmer(reflected_variable ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)
model_gamma2 <- glmer(reflected_variable ~ treat_list+day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)


anova(model_log, model_sqrt)
anova(model, model_sqrt)
anova(model_gamma, model_sqrt)
anova(model_gamma, model_gamma2)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma2
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=2)

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0),
  "PathogenMinusShamMON"=c(0,-1,0))
differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
print(rbind(cd_wed, cd_mon))
##
DF$variable<-DF$heterogeneity_list
hist(DF$variable)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), data = DF)
model_gamma <- glmer(variable ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)

anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=3)

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
print(rbind(cd_wed, cd_mon))

##
DF$variable<-DF$modularity_list
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable+0.2) ~ treat_list*day_list+  (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable+0.2) ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), data = DF)
model_gamma <- glmer(variable+0.2 ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)

anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_log
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=3)
mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list','PATHOGEN', 'SHAM')
print(rbind(cd_wed, cd_mon))
cd_mon
#CLUSTERING
DF$variable<-DF$clustering_list
hist(DF$variable)
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+  (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), data = DF)
model_gamma <- glmer(variable ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)
C <- max(DF$variable) + 1
DF$reflected_variable <- C - DF$variable
model_gamma2 <- glmer(reflected_variable ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)



anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma2
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=3)
mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))
differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list','PATHOGEN', 'SHAM')
print(rbind(cd_wed, cd_mon))



#NO DOL
DF <- read.csv('/home/ll16598/Documents/ARCHITECTURAL_IMMUNITY/ABM_CODE/0_combined_social_analysis.csv', header=TRUE)

DF$day_list


##DENSITY
DF$variable<-DF$density_list
hist(DF$variable)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt2 <- lmer(sqrt(variable) ~ treat_list+day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_gamma <- glmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), family='Gamma', data = DF)
C <- max(DF$variable) + 1
DF$reflected_variable <- C - DF$variable
hist(DF$reflected_variable)
model_gamma <- glmer(reflected_variable ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)
model_gamma2 <- glmer(reflected_variable ~ treat_list+day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)


anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model_gamma2)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_sqrt
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=2)


mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')

cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
print(rbind(cd_wed, cd_mon))


###Efficiency
DF$variable<-DF$efficiency_list
hist(DF$variable)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|iteration)+(1|t_chunks), data = DF)
model_sqrt2 <- lmer(sqrt(variable) ~ treat_list+day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
C <- max(DF$variable) + 1
DF$reflected_variable <- C - DF$variable

hist(DF$reflected_variable)
model_gamma <- glmer(reflected_variable ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)
model_gamma2 <- glmer(reflected_variable ~ treat_list+day_list+ (1|iteration/colony_list/subset_list/t_chunks), family='Gamma', data = DF)


anova(model_log, model_sqrt)
anova(model, model_sqrt)
anova(model_gamma, model_sqrt)
anova(model_gamma, model_gamma2)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma2
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=2)

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0),
  "PathogenMinusShamMON"=c(0,-1,0))
differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')

cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
print(rbind(cd_wed, cd_mon))
##
DF$variable<-DF$heterogeneity_list
hist(DF$variable)
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), data = DF)
model_gamma <- glmer(variable ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)

anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=3)

mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'SHAM', 'PATHOGEN')
print(rbind(cd_wed, cd_mon))

##
DF$variable<-DF$modularity_list
MON_DF<-subset(DF, day_list=='MON')
WED_DF<-subset(DF, day_list=='WED')
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable+0.2) ~ treat_list*day_list+  (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable+0.2) ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), data = DF)
model_gamma <- glmer(variable+0.2 ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)

anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_log
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=3)
mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))

differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
print(rbind(cd_wed, cd_mon))
cd_mon
#CLUSTERING
DF$variable<-DF$clustering_list
hist(DF$variable)
model <- lmer(variable ~ treat_list*day_list+ (1|colony_list/subset_list)+(1|week_list)+(1|t_chunks), data = DF)
model_log <- lmer(log(variable) ~ treat_list*day_list+  (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list)+(1|t_chunks), data = DF)
model_sqrt <- lmer(sqrt(variable) ~ treat_list*day_list+ (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), data = DF)
model_gamma <- glmer(variable ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)
C <- max(DF$variable) + 1
DF$reflected_variable <- C - DF$variable
model_gamma2 <- glmer(reflected_variable ~ treat_list*day_list + (1|iteration/colony_list/subset_list/t_chunks)+(1|week_list), family='Gamma', data = DF)



anova(model_log, model_sqrt)
anova(model, model_log)
anova(model_gamma, model_sqrt)
anova(model_gamma, model)
anova(model_sqrt, model_sqrt2)
par(mfrow=c(2,2))
qq_plot(model, 'model')
qq_plot(model_log, 'model_log')
qq_plot(model_sqrt, 'model_sqrt')
qq_plot(model_gamma, 'model_gamma')
model1<-model_gamma2
resid<-residuals(model1)
shapiro.test(resid)
hist(resid)
summary(model1)
car::Anova(model1, type=3)
mat <-rbind(
  "PathogenMinusShamWED"=c(0,-1,0,-1),
  "PathogenMinusShamMON"=c(0,-1,0,0))
differences <-glht(model1, linfct = mat, correction='BH')
summary(differences)
cd_wed <- cohens_d(WED_DF, 'variable', 'treat_list', 'PATHOGEN', 'SHAM')
cd_mon <- cohens_d(MON_DF, 'variable', 'treat_list','PATHOGEN', 'SHAM')
print(rbind(cd_wed, cd_mon))
