library(lme4)

ex2 <- read.csv('../data/experiments/stats_comprehension.csv')

# Make condition into a factor with the levels organzied as 1D, 1D, 2D
ex2$condition <- factor(ex2$condition, levels=c('angle', 'size', 'both'))

# Helmert contrast coding
contrasts(ex2$condition) <- contr.helmert(3)

# Run model with correctness predicted by condition with random intercepts for subject
# Family = binomial because correctness is binary
model <- glmer(correct ~ condition + (1|subject), data=ex2, family=binomial)

# "Condition 1" is difference between (Angle-only) and (Size-only)
# "Condition 2" is difference between (both 1D conditions) and (Angle & Size)
summary(model)