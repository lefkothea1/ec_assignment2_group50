### Statistical Analysis Gain (Group A x Group A, CMA x MO-CMA)

## read in data


c2gain <- scan(text = "
-323.99999999999955
-149.63999999999947
-186.99999999999937
-371.23999999999967
-346.7999999999995
-314.1999999999995
-111.27999999999938
-92.87999999999948
-279.7999999999995
-284.59999999999974")

c3gain <- scan(text = "
-243.91999999999945
-420.87999999999977
-326.15999999999974
-346.9999999999998
-273.63999999999976
-416.79999999999984
-250.5199999999998
-293.83999999999946
-368.91999999999973
-238.67999999999944")

a2gain <- scan(text = "
-59.479999999999094
-261.9199999999996
-217.19999999999942
-268.59999999999945
-231.15999999999948
-263.5999999999994
-198.55999999999932
-338.59999999999945
-117.8799999999991
-299.3199999999994")

a3gain <- scan(text = "
-356.87999999999977
-365.6799999999998
-241.3599999999995
-487.5999999999998
-325.59999999999985
-160.43999999999943
-242.63999999999953
-350.11999999999966
-347.99999999999983
-156.95999999999918")

#dataframe

dat <- data.frame(group = rep(c("A","B"), each = 10),
                  algo = rep(c("CMA", "MO-CMA"), each = 20),
                  gain = c(c2gain, c3gain, a2gain, a3gain))

dat$group <- as.factor(dat$group)
dat$algo <- as.factor(dat$algo)

str(dat)

## tests:

t.test(c2gain, a2gain)
t.test(c3gain, a3gain)
t.test(c2gain, c3gain)
t.test(a2gain, a3gain)


## table of descriptive statistics

table_max <- aggregate(gain ~ algo + group, dat, FUN = max)
table_mean <- aggregate(gain ~ algo + group, dat, FUN = mean)
table_min <- aggregate(gain ~ algo + group, dat, FUN = min)
