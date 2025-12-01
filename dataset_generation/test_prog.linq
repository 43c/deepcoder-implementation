a <- [int]
b <- ZIPWITH - a a
c <- MAP INC b
d <- MAP SHL c
e <- SCANL1 * d
f <- REVERSE e
g <- MAP SHR f
h <- ZIPWITH * a g
i <- SCANL1 + h
j <- LAST i