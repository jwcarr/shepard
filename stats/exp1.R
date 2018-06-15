library(lme4)
packageVersion("lme4")

exp1 <- read.csv('../data/experiments/exp1_stats.csv')

# Make Test Type into a factor
exp1$test_type <- factor(exp1$test_type, levels=c('production', 'comprehension'))

# Make Category System into a factor with the levels organzied as 1D, 1D, 2D
exp1$category_system <- factor(exp1$category_system, levels=c('angle', 'size', 'both'))

# Helmert contrast coding between different category systems
contrasts(exp1$category_system) <- contr.helmert(3)

# Correctness predicted by Test Type and Category System with random intercepts for subject
# Family = binomial because correctness is binary
model <- glmer(correct ~ test_type * category_system + (1|subject), data=exp1, family=binomial)

# "category_system1" is difference between (Angle-only) and (Size-only)
# "category_system2" is difference between (Angle-only and Size-only) and (Angle & Size)
summary(model)
