library(lme4)
packageVersion("lme4")

exp2 <- read.csv('../data/experiments/exp2_stats.csv')

exp2$chain <- factor(exp2$chain) # Make chain a factor
exp2$generation <- exp2$generation - 1 # Start generation at 0 so that the intercept is meaningful

# Expressivity
model <- lmer(expressivity ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(expressivity ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)

# Transmission error
model <- lmer(transmission_error ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(transmission_error ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)

# Complexity
model <- lmer(complexity ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(complexity ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)

# Communicative cost
model <- lmer(communicative_cost ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(communicative_cost ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)
