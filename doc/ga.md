# GA

## Steps for standard GAs

1. Choose initial population
2. Assign a fitness calculation
3. Perform elitism (if elitism is to be applied)
4. Selection
5. Mating & crossover
6. Mutation

## Real coded GAs

Disadvntage of binary coded GA:

* more computation
* lower accuracy
* longer computing time
* solution space discontinuity
* hamming cliff

## Linear arithmetic crossover

\small

* For two parents:

  $X_1: {x_1^1, x_1^2, ... , x_1^k, x_1^{k+1}, ..., x_1^n}$
  
  $X_2: {x_2^1, x_2^2, ... , x_2^k, x_2^{k+1}, ..., x_2^n}$

  Peak random gene at ${k}$. Then the two children become:

  $X_1: {x_1^1, ..., x_1^k, 0.5 x_2^k + 0.5 x_1^k, ..., x_1^n}$
  
  $X_2: {x_1^1, ..., x_1^k, 1.5 x_2^k - 0.5 x_1^k, ..., x_1^n}$

  $X_3: {x_1^1, ..., x_1^k, -0.5 x_2^k + 1.5 x_1^k, ..., x_1^n}$

Which from these three, the best two are selected.

## Single arithmetic crossover

\small

* For two parents:

  $X_1: {x_1^1, x_1^2, ... , x_1^k, x_1^{k+1}, ..., x_1^n}$
  
  $X_2: {x_2^1, x_2^2, ... , x_2^k, x_2^{k+1}, ..., x_2^n}$

  Peak random gene at ${k}$. Then the two children become:

  $X_1: {x_1^1, x_1^2, ... , \alpha x_2^k + (1 - \alpha)x_1^k, x_1^{k+1}, ..., x_1^n}$
  
  $X_2: {x_2^1, x_2^2, ... , \alpha x_1^k + (1 - \alpha)x_2^k, x_2^{k+1}, ..., x_2^n}$

## Simple arithmetic crossover

\small

* For two parents:

  $X_1: {x_1^1, x_1^2, ... , x_1^k, x_1^{k+1}, ..., x_1^n}$

  $X_2: {x_2^1, x_2^2, ... , x_2^k, x_2^{k+1}, ..., x_2^n}$

  Peak random gene at ${k}$. Then the two children become:

  $X_1: {x_1^1, x_1^2, ... , x_1^k, \alpha x_2^{k+1} + (1 - \alpha)x_1^{k+1}, ..., \alpha x_2^{n} + (1 - \alpha)x_1^{n}}$
  
  $X_2: {x_2^1, x_2^2, ... , x_2^k, \alpha x_1^{k+1} + (1 - \alpha)x_2^{k+1}, ..., \alpha x_1^{n} + (1 - \alpha)x_2^{n}}$

## Whole arithmetic crossover

\small

* For two parents:

  $X_1: {x_1^1, x_1^2, ... , x_1^k, x_1^{k+1}, ..., x_1^n}$

  $X_2: {x_2^1, x_2^2, ... , x_2^k, x_2^{k+1}, ..., x_2^n}$

  Then the two children become:

  $X_1: {\alpha x_1^1 + (1 - \alpha)x_2^1, ..., \alpha x_1^k + (1 - \alpha)x_2^k, ..., \alpha x_1^n + (1 - \alpha)x_2^n}$
  
  $X_2: {\alpha x_2^1 + (1 - \alpha)x_1^1, ..., \alpha x_2^k + (1 - \alpha)x_1^k, ..., \alpha x_2^n + (1 - \alpha)x_1^n}$
