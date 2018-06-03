library(lme4)

ex3 <- read.csv('../data/experiments/stats_iterated_learning.csv')

ex3$chain <- factor(ex3$chain) # Make chain a factor
ex3$generation <- ex3$generation - 1 # Start generation at 0 so that the intercept is meaningful

# Expressivity
model <- lmer(expressivity ~ generation + (1+generation|chain), data=ex3, REML=T)
null_model <- lmer(expressivity ~ 1 + (1+generation|chain), data=ex3, REML=F)
summary(model)
anova(null_model, model)

# Transmission error
model <- lmer(transmission_error ~ generation + (1+generation|chain), data=ex3, REML=T)
null_model <- lmer(transmission_error ~ 1 + (1+generation|chain), data=ex3, REML=F)
summary(model)
anova(null_model, model)

# Complexity
model <- lmer(complexity ~ generation + (1+generation|chain), data=ex3, REML=T)
null_model <- lmer(complexity ~ 1 + (1+generation|chain), data=ex3, REML=F)
summary(model)
anova(null_model, model)

# Communicative cost
model <- lmer(communicative_cost ~ generation + (1+generation|chain), data=ex3, REML=T)
null_model <- lmer(communicative_cost ~ 1 + (1+generation|chain), data=ex3, REML=F)
summary(model)
anova(null_model, model)
