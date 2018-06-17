library(lme4)
packageVersion("lme4")

exp2 <- read.csv('../data/experiments/exp2_stats.csv')

# Make chain into a factor
exp2$chain <- factor(exp2$chain)

# Start generations at 0 so that the intercept is meaningful (gen1 becomes gen0, etc.)
exp2$generation <- exp2$generation - 1

# Expressivity predicted by generation with random intercepts for chain (and by-chain random slopes for generation)
model <- lmer(expressivity ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(expressivity ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)

# Transmission error predicted by generation with random intercepts for chain (and by-chain random slopes for generation)
model <- lmer(error ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(error ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)

# Complexity predicted by generation with random intercepts for chain (and by-chain random slopes for generation)
model <- lmer(complexity ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(complexity ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)

# Communicative Cost predicted by generation with random intercepts for chain (and by-chain random slopes for generation)
model <- lmer(cost ~ generation + (1+generation|chain), data=exp2, REML=T)
null_model <- lmer(cost ~ 1 + (1+generation|chain), data=exp2, REML=F)
summary(model)
anova(null_model, model)
