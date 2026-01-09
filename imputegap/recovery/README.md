<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegap_patterns_b.png" >

<br />

# Patterns

All missingness patterns developed in ImputeGAP are available in the `ts.patterns` module.

To list all the available patterns, you can use this command:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"Missingness patterns : {ts.patterns}")
```

## Setup

> - `M`: number of time series  
> - `N`: length of time series  
> - `R`: user-defined rate of missing values (%); default = 0.2
> - `W`: user-defined offset window in the beginning of the series (%); default = 0.1
> - `S`: user-defined rate of contaminated series (%); default = 0.2

## MONO-BLOCK

**One missing block per series**

### Aligned
The missing blocks are aligned. 

> - `R ∈ [1%, (100-W)%]`  
> - The size of a single missing block varies between 1% and (100 - `W`)% of `N`.  
> - The starting position is the same and begins at `W` and progresses until the size of the missing block is reached, affecting the first series from the top up to `S%` of the dataset.

### Disjoint
The missing blocks are disjoint.  


> - `R ∈ [1%, (100-W)%]`  
> - The size of a single missing block varies between 1% and (100 - `W`)% of `N`.  
> - The starting position of the first missing block begins at `W`.  
> - Each subsequent missing block starts immediately after the previous one ends, continuing this pattern until the limit of the dataset or `N` is reached.

### Overlap
The missing blocks are overlapping.  

> - `R ∈ [1%, (100-W)%]`  
> - The size of a single missing block varies between 1% and (100 - `W`)% of `N`.  
> - The starting position of the first missing block begins at `W`.  
> - Each subsequent missing block starts after the previous one ends, but with a shift back of `X%`, creating an overlap.  
> - This pattern continues until the limit or `N` is reached.

### Scattered
The missing blocks are scattered.  

> - `R ∈ [1%, (100-W)%]`  
> - The size of a single missing block varies between 1% and (100 - `W`)% of `N`.  
> - The starting position is randomly shifted by adding a random value to `W`, then progresses until the size of the missing block is reached, affecting the first series from the top up to `S%` of the dataset.

## MULTI-BLOCK

**Multiple missing blocks per series**

### MCAR
The blocks are missing completely at random.  

> - `R ∈ [1%, (100-W)%]`  
> - Data blocks of the same size are removed from arbitrary series at a random position between `W` and `N`, until the total number of missing values per series is reached.

### Block Distribution
The missing blocks follow a distribution. 

> - `R ∈ [1%, (100-W)%]`  
> - Data is removed following a distribution given by the user for every value of the series, affecting the first series from the top up to `S%` of the dataset.  

To configure the distribution pattern, please refer to this [tutorial](https://imputegap.readthedocs.io/en/latest/tutorials_distribution.html).