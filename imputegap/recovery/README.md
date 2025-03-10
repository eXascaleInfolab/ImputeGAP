<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# CONTAMINATION
## Patterns
<table>
    <tr>
        <td>M</td><td>Number of time series</td>
    </tr>
    <tr>
        <td>N</td><td>Lentgh of time series</td>
    </tr>
    <tr>
        <td>P</td><td>Starting position (protection)</td>
    </tr>
    <tr>
        <td>R</td><td>Missing rate of the pattern</td>
    </tr>
    <tr>
        <td>S</td><td>percentage of series selected</td>
    </tr>
    <tr>
        <td>W</td><td>Total number of values to remove</td>
    </tr>
    <tr>
        <td>B</td><td>Block size</td>
    </tr>
</table><br />

---

### MCAR (MULTI-BLOCK)
MCAR selects random series and remove block at random positions until a total of W of all points of time series are missing.
This pattern uses random number generator with fixed seed and will produce the same blocks every run.

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R</td>
    </tr>
    <tr>
        <td>B</td><td>2 - 20</td>
    </tr>
 </table>

<i>Pattern MCAR : dataset_rate=0.4, series_rate=0.4, offset=0.1</i><br />
![pattern MCAR](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/mcar.jpg)

---

<br />


### GAUSSIAN (MULTI-BLOCK)
The **GAUSSIAN** pattern introduces missing values into a percentage of time series, determined based on probabilities derived from a Gaussian distribution.

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R * probability</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table>

<i>Pattern GAUSSIAN : dataset_rate=0.4, series_rate=0.6, std_dev=0.5, offset=0.1</i><br />
![pattern GAUSSIAN](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/gaussian.jpg)


---


<br />


### DISTRIBUTION (MULTI-BLOCK)
The **DISTRIBUTION** pattern introduces missing values into a percentage of time series, determined based on probabilities given by the user (from any source or library).

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R * probability</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table>

<i>Pattern DISTRIBUTION : dataset_rate=0.4, series_rate=0.6, std_dev=0.5, offset=0.1</i><br />
![pattern DISTRIBUTION](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/gaussian.jpg)


---

<br />


### MISSING PERCENTAGE (MONO-BLOCK)
**MISSING PERCENTAGE** selects a percentage of time series to contaminate, applying the desired percentage of missing values from the beginning to the end of each selected series.



<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table>



<i>Pattern MISSING PERCENTAGE : dataset_rate=0.4, series_rate=0.6, std_dev=0.5, offset=0.1</i><br />
![pattern MISSING PERCENTAGE](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/mp.jpg)


---

<br />


### BLACKOUT (MONO-BLOCK)
The **BLACKOUT** pattern introduces missing values across all time series by removing a specified percentage of data points from each series, creating uniform gaps for analysis.


<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table>


<i>Pattern BLACKOUT : dataset_rate=0.4, series_rate=0.6, std_dev=0.5, offset=0.1</i><br />
![pattern BLACKOUT](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/blackout.jpg)


---



<br />

### DISJOINT (MONO-BLOCK)
The **DISJOINT** pattern introduces missing values into time series by selecting segments with non-overlapping intervals. This process continues until either the missing rate limit is reached or the series length is exhausted.

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table>

<i>Pattern DISJOINT :  series_rate=0.1, limit=1, offset=0.1</i><br />
![pattern DISJOINT](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/disjoint.jpg)

---

<br />

### OVERLAP (MONO-BLOCK)
The **OVERLAP** pattern selects time series segments for introducing missing values by using a disjoint interval that is shifted by a specified percentage. This process continues until either the missing rate limit is reached or the series length is exhausted.


<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>N * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table>

<i>Pattern OVERLAP :  series_rate=0.1, shift=0.05, limit=1, offset=0.1</i><br />
![pattern OVERLAP](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/dataset/docs/patterns/overlap.jpg)

---

<br />
