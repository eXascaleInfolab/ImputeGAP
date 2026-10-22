<img align="right" width="140" height="140" src="https://raw.githubusercontent.com/eXascaleInfolab/ImputeGAP/main/assets/logo_imputegab.png">
<br /> <br />

# Welcome to ImputeGAP

ImputeGAP is a comprehensive Python library for imputation of missing values in time series data. It implements user-friendly APIs to easily visualize, analyze, and repair incomplete time series datasets. The library supports a diverse range of imputation algorithms and modular missing data simulation catering to datasets with varying characteristics. ImputeGAP includes extensive customization options, such as automated hyperparameter tuning, benchmarking, explainability, and downstream evaluation.

In detail, the package provides:

  - Over 40 state-of-the-art time series imputation algorithms from six different families ([Algorithms](#Available-Imputation-Algorithms))
  - Several imputation univariate time series datasets and utilities to handle multivariate ones ([Datasets](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets))
  - Configurable contamination module that simulates real-world missingness patterns ([Patterns](https://imputegap.readthedocs.io/en/latest/missingness_patterns.html))
  - AutoML techniques to parameterize the imputation algorithms ([AutoML](#Parameter-Tuning))
  - Unified benchmarking pipeline to evaluate the performance of imputation algorithms ([Benchmark](#benchmark))
  - Modular analysis tools to assess the impact of imputation on time series downstream tasks ([Downstream](#downstream))
  - Expainability module to understand the impact of time series features on the imputation results ([Explainer](#explainer))
  - Adjustable wrappers to integrate new algorithms in different languages: Python, C++, Matlab, Java, and R ([Contributing](https://imputegap.readthedocs.io/en/latest/contributing.html))

<br>

![Python](https://img.shields.io/badge/Python-v3.13-blue) ![Release](https://img.shields.io/badge/Release-v1.1.2-brightgreen)  ![Coverage](https://img.shields.io/badge/Coverage-94%25-brightgreen) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-informational) [![Docs](https://img.shields.io/badge/Docs-available-brightgreen?style=flat&logo=readthedocs)](https://imputegap.readthedocs.io/)

<i>If you like our library, please add a ⭐ in our GitHub repository.</i>

<br>

| Tools                | URL                                                                                      |
|----------------------|------------------------------------------------------------------------------------------|
| 📚 **Documentation** | [https://imputegap.readthedocs.io/](https://imputegap.readthedocs.io/)                   |
| 📦 **PyPI**          | [https://pypi.org/project/imputegap/](https://pypi.org/project/imputegap/)               |
| 📁 **Datasets**      | [Description](https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets) |


- ---


# Available Imputation Algorithms
| **Algorithm**             | **Family**        | **Venue -- Year**            |
|---------------------------|-------------------|------------------------------|
| MOMENT [[39]](#ref39)     | LLMs              | ICML -- 2024                 |
| NuwaTS [[35]](#ref35)     | LLMs              | Arxiv -- 2024                |
| GPT4TS [[36]](#ref36)     | LLMs              | NeurIPS -- 2023              |
| 🚧 UniTS [[44]](#ref44)   | LLMs              | NeurIPS -- 2024              |
| 🚧 Timer [[45]](#ref45)   | LLMs              | ICML -- 2024                 |
| MissNet [[27]](#ref27)    | Deep Learning     | KDD -- 2024                  |
| BayOTIDE [[30]](#ref30)   | Deep Learning     | ICML -- 2024                 |
| BitGraph [[32]](#ref32)   | Deep Learning     | ICLR -- 2024                 |
| MPIN [[25]](#ref25)       | Deep Learning     | PVLDB -- 2023                |
| TimesNet [[37]](#ref37)   | Deep Learning     | ICLR -- 2023                 |
| SAITS [[41]](#ref41)      | Deep Learning     | ESWA -- 2023                 |
| PRISTI [[26]](#ref26)     | Deep Learning     | ICDE -- 2023                 |
| GRIN [[29]](#ref29)       | Deep Learning     | ICLR -- 2022                 |
| CSDI [[38]](#ref38)       | Deep Learning     | NeurIPS -- 2021              |
| HKMFT [[31]](#ref31)      | Deep Learning     | TKDE -- 2021                 |
| DeepMVI [[24]](#ref24)    | Deep Learning     | PVLDB -- 2021                |
| MRNN [[22]](#ref22)       | Deep Learning     | IEEE Trans on BE -- 2019     |
| BRITS [[23]](#ref23)      | Deep Learning     | NeurIPS -- 2018              |
| GAIN [[28]](#ref28)       | Deep Learning     | ICML -- 2018                 |
| 🚧 SSGAN [[40]](#ref40)   | Deep Learning     | AAAI -- 2021                 |
| 🚧 GP-VAE [[42]](#ref42)  | Deep Learning     | AISTATS -- 2020              |
| 🚧 NAOMI [[43]](#ref43)   | Deep Learning     | NeurIPS -- 2019              |
| CDRec [[1]](#ref1)        | Matrix Completion | KAIS -- 2020                 |
| TRMF [[8]](#ref8)         | Matrix Completion | NeurIPS -- 2016              |
| ROSL [[4]](#ref4)         | Matrix Completion | CVPR -- 2014                 |
| GROUSE [[3]](#ref3)       | Matrix Completion | Arxiv -- 2010                |
| SoftImpute [[6]](#ref6)   | Matrix Completion | JMLR -- 2010                 |
| SVT [[7]](#ref7)          | Matrix Completion | SIAM J. OPTIM -- 2010        |
| SPIRIT [[5]](#ref5)       | Matrix Completion | VLDB -- 2005                 |
| IterativeSVD [[2]](#ref2) | Matrix Completion | BIOINFORMATICS -- 2001       |
| TKCM [[11]](#ref11)       | Pattern Search    | EDBT -- 2017                 |
| STMVL [[9]](#ref9)        | Pattern Search    | IJCAI -- 2016                |
| DynaMMo [[10]](#ref10)    | Pattern Search    | KDD -- 2009                  |
| IIM [[12]](#ref12)        | Machine Learning  | ICDE -- 2019                 |
| XGBOOST [[13]](#ref13)    | Machine Learning  | KDD -- 2016                  |
| MissForest [[15]](#ref15) | Machine Learning  | BioInformatics -- 2012       |
| MICE [[14]](#ref14)       | Machine Learning  | Statistical Software -- 2011 |
| KNNImpute                 | Statistics        | -                            |
| Interpolation             | Statistics        | -                            |
| MinImpute                 | Statistics        | -                            |
| ZeroImpute                | Statistics        | -                            |
| MeanImpute                | Statistics        | -                            |
| MeanImputeBySeries        | Statistics        | -                            |
---

### **Quick Navigation**

- **Getting Started**  
  - [System Requirements](#system-requirements)  
  - [Installation](#installation)  

- **Code Snippets**  
  - [Dataset Loading](#dataset-loading)  
  - [Contamination](#contamination)  
  - [Imputation](#imputation)  
  - [Auto-ML](#Parameter-Tuning)  
  - [Explainer](#explainer)  
  - [Downstream Evaluation](#downstream)
  - [Benchmark](#benchmark)  
  - [Notebooks](#Jupyter-Notebooks)  

- **Contribute**  
  - [Integration Guide](#contribution)  

- **Additional Information**  
   - [Maintainers](#maintainers)  
   - [References](#references)  
 

---

<br> <br>

# Getting Started

## System Requirements

ImputeGAP is compatible with Python>=3.11 and Unix-compatible environment.

<i>To create and set up an environment with Python 3.12, please refer to the [installation guide](https://imputegap.readthedocs.io/en/latest/getting_started.html).</i>


<br>

## Installation


### pip

To install/update the latest version of ImputeGAP, run the following command:

```bash
pip install imputegap
``` 

<br>

### Source

If you would like to extend the library, you can install from source:

```bash
git init
git clone https://github.com/eXascaleInfolab/ImputeGAP
cd ImputeGAP/
source imputegap_source.sh
```

<br>

### Docker

Alternatively, you can download the latest version of ImputeGAP with all dependencies pre-installed using Docker.

Launch Docker and make sure it is running:

```bash
docker version
``` 

Pull the ImputeGAP Docker image (add `--platform linux/x86_64` in the command for MacOS) :

```bash
docker pull qnater/imputegap:1.1.21
```

Run the Docker container:

```bash
docker run -p 8888:8888 qnater/imputegap:1.1.21
``` 


---


<br> <br>

# Tutorials

## Dataset Loading

ImputeGAP comes with several time series datasets. The list of datasets is described [here](https://imputegap.readthedocs.io/en/latest/datasets.html).

As an example, we use the eeg-alcohol dataset, composed of individuals with a genetic predisposition to alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz. The dimensions of the dataset are 64 series, each containing 256 values.


### Example Loading
You can find this example of normalization in the file [`runner_loading.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_loading.py).

To load and plot the eeg-alcohol dataset from the library:

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset from the library
ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

# print and plot a subset of time series
ts.print(nbr_series=4, nbr_val=4)
ts.plot(input_data=ts.data, nbr_series=6, nbr_val=100, save_path="./imputegap_assets")
```

The module ``ts.datasets`` contains all the publicly available datasets provided by the library, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"ImputeGAP datasets : {ts.datasets}")
```

---

## Contamination
We now describe how to simulate missing values in the loaded dataset. ImputeGAP implements eight different missingness patterns. For more details about the patterns, please refer to the documentation on this [page](https://imputegap.readthedocs.io/en/latest/missingness_patterns.html).
<br></br>

### Example Contamination
You can find this example in the file [`runner_contamination.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_contamination.py).

ImputeGAP implements a module to simulate different missingness patterns in time series. The GenGap module can be invoked externally. The patterns are described [here](https://imputegap.readthedocs.io/en/latest/missingness_patterns.html)

As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

```python
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

# contaminate the time series with MCAR pattern
ts_m = GenGap.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

# plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")
```

All missingness patterns developed in ImputeGAP are available in the ``ts.patterns`` module. They can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"Missingness patterns : {ts.patterns}")
```



---

## Imputation

In this section, we will illustrate how to impute the contaminated time series. Our library implements six families of imputation algorithms: Statistical, Machine Learning, Matrix Completion, Deep Learning, Pattern Search, and Large Language Models.
The list of algorithms is described [here](https://imputegap.readthedocs.io/en/latest/algorithms.html).

### Example Imputation
You can find this example in the file [`runner_imputation.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_imputation.py).

Let's illustrate the imputation using the CDRec algorithm from the Matrix Completion family.

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

# contaminate the time series
ts_m = GenGap.mcar(ts.data)

# impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")
```

Imputation can be performed using either default values or user-defined values. To specify the parameters, please use a dictionary in the following format:

```python
imputer.impute(params={"rank": 5, "epsilon": 0.01, "iteration": 100})
```

All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module, which can be listed as follows:
```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"Imputation families : {ts.families}")
print(f"Imputation algorithms : {ts.algorithms}")
```

---


## Parameter Tuning
The Optimizer component manages algorithm configuration and hyperparameter tuning. The parameters are defined by providing a dictionary containing the ground truth, the chosen optimizer, and the optimizer's options. Several search algorithms are available, including those provided by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

### Example Auto-ML
You can find this example in the file [`runner_optimization.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_optimization.py).

Let's illustrate the imputation using the CDRec algorithm and Ray-Tune AutoML:

```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

# contaminate and impute the time series
ts_m = GenGap.mcar(ts.data)
imputer = Imputation.MatrixCompletion.CDRec(ts_m)

# use Ray Tune to fine tune the imputation algorithm
imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

# compute the imputation metrics with optimized parameter values
imputer.score(ts.data, imputer.recov_data)

# compute the imputation metrics with default parameter values
imputer_def = Imputation.MatrixCompletion.CDRec(ts_m).impute()
imputer_def.score(ts.data, imputer_def.recov_data)

# print the imputation metrics with default and optimized parameter values
ts.print_results(imputer_def.metrics, text="Default values")
ts.print_results(imputer.metrics, text="Optimized values")

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")

# save hyperparameters
utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")
```


---



## Benchmark

ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms[[33]](#ref33) . Users have full control over the benchmark by customizing various parameters,  including the list of the algorithms to compare, the optimizer, the datasets to evaluate, the missingness patterns, the range of missing values, and the performance metrics.


### Example Benchmark
You can find this example in the file [`runner_benchmark.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_benchmark.py).

The benchmarking module can be utilized as follows:

```python
from imputegap.recovery.benchmark import Benchmark

# launch the evaluation
bench = Benchmark()
bench.eval_imputation(algorithms=["MeanImpute", "SoftImpute"],
           datasets=["eeg-alcohol"],
           patterns=["mcar"],
           x_axis=[0.1, 0.2, 0.4, 0.6, 0.8])

# display the plot
bench.subplots.show()
```

---


## Downstream
ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional applications in the future.

### Example Downstream
You can find this example in the file [`runner_down_imputation.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_down_imputation.py).

Below is an example of how to call the  process for the model Prophet by defining a dictionary for the evaluator and selecting the model:
downstream
```python
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()

# load and normalize the timeseries
ts.load_series(utils.search_path("forecast-economy"), normalizer="z_score")

# contaminate the time series
ts_m = GenGap.aligned(ts.data, rate_series=0.8)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the downstream results
downstream_config = {"task": "forecast", "model": "hw-add", "baseline": "ZeroImpute"}
imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
ts.print_results(imputer.downstream_metrics, text="Downstream results")
```

All downstream models developed in ImputeGAP are available in the ``ts.forecasters`` module, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries

ts = TimeSeries()
print(f"ImputeGAP downstream models for forecasting : {ts.forecasters}")
```


---


## Explainer

The library provides insights into the algorithm’s behavior by identifying the features that impact the imputation results. It trains a regression model to predict imputation results across various methods and uses SHapley Additive exPlanations ([SHAP](https://shap.readthedocs.io/en/latest/)) to reveal how different time series features influence the model’s predictions.


### Example Explainer
You can find this example in the file [`runner_explainer.py`](https://github.com/eXascaleInfolab/ImputeGAP/blob/main/imputegap/runner_explainer.py).

Let’s illustrate the explainer using the CDRec algorithm and MCAR missingness pattern:


```python
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.tools import utils

# initialize the time series and explainer object
ts = TimeSeries()
exp = Explainer()

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

# configure the explanation
exp.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar", file_name=ts.name, algorithm="CDRec")

# print the impact of each feature
exp.print(exp.shap_values, exp.shap_details)

# plot the features impact
exp.show()
```

All feature extractors developed in ImputeGAP are available in the ``ts.extractors`` module, which can be listed as follows:

```python
from imputegap.recovery.manager import TimeSeries
ts = TimeSeries()
print(f"ImputeGAP features extractors : {ts.extractors}")
```


---

## Jupyter Notebooks

ImputeGAP provides Jupyter notebooks available through the following links:

- [Jupyter: Imputegap Imputation Pipeline Notebook](https://github.com/eXascaleInfolab/ImputeGAP/blob/refs/heads/main/imputegap/notebooks/01_imputegap_pipeline_creation.ipynb)
- [Jupyter: Imputegap Advanced Analysis Notebook](https://github.com/eXascaleInfolab/ImputeGAP/blob/refs/heads/main/imputegap/notebooks/02_imputegap_advanced_analytics.ipynb)



## Google Colab Notebooks

ImputeGAP provides Google Colab notebooks available through the following links:

- [Google Colab: Imputegap Imputation Pipeline Notebook](https://colab.research.google.com/drive/1Kq1_HVoCTWLtB1zyryR35opxXmaprztV?usp=sharing)
- [Google Colab: Imputegap Advanced Analysis Notebook](https://colab.research.google.com/drive/1iOzLtpZTA3KDoyIc-srw2eoX5soEmP8x?usp=sharing)



---

## Contribution
To add your own imputation algorithm, please refer to the detailed [integration guide](https://imputegap.readthedocs.io/en/latest/contributing.html).


---


<br> <br>

# Citing

If you use ImputeGAP in your research, please cite these papers:

```
@article{nater2025imputegap,
  title = {ImputeGAP: A Comprehensive Library for Time Series Imputation},
  author = {Nater, Quentin and Khayati, Mourad and Pasquier, Jacques},
  year = {2025},
  eprint = {2503.15250},
  archiveprefix = {arXiv},
  primaryclass = {cs.LG},
  url = {https://arxiv.org/abs/2503.15250}
}
```

<br>

```
@article{nater2025kdd,
  title = {A Hands-on Tutorial on Time Series Imputation with ImputeGAP},
  author = {Nater, Quentin and Khayati, Mourad and Cudré-Mauroux, Philippe},
  year = {2025},
  booktitle = {SIGKDD Conference on Knowledge Discovery and Data Mining (To Appear)},
  series = {KDD2025}
}
```

<br> <br>

---

## Maintainers

<table>
  <tr>
    <td width="50%">
      <div  style="text-align: center;" >
        <a href="https://exascale.info/members/quentin-nater/">
            <img src="https://imputegap.readthedocs.io/en/latest/_images/quentin_nater.png" alt="Quentin Nater" width="150"/>
        </a><br>
        <strong>Quentin Nater</strong>
      </div>
      <br><br>
      <div style="text-align: justify;"> 
        Quentin is a PhD student jointly supervised by
        <a href="https://exascale.info/members/mourad-khayati/">Mourad Khayati</a> and
        <a href="https://exascale.info/phil/">Philippe Cudré-Mauroux</a> at the Department of Computer Science of the
        <a href="https://www.unifr.ch/home/en/">University of Fribourg</a>, Switzerland. He completed his Master’s degree in Digital Neuroscience at the University of Fribourg.
        His research focuses on <strong>time series analytics</strong>,
        including <strong>data imputation</strong>, <strong>machine learning</strong>, and
        <strong>multimodal learning</strong>.
      </div><br>
      👉 <a href="https://exascale.info/members/quentin-nater/">Home Page</a>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td width="50%" >
      <div  style="text-align: center;" >
        <a href="https://mkhayati.github.io/">
          <img src="https://imputegap.readthedocs.io/en/latest/_images/mourad_khayati.png" alt="Mourad Khayati" width="150" />
        </a><br>
        <strong>Mourad Khayati</strong>
      </div>
      <br><br>
      <div style="text-align: justify;"> 
        Mourad is a Senior Researcher and Lecturer with the
        <a href="https://exascale.info/">eXascale Infolab</a> and the Advanced Software Engineering group at the Department of Computer Science of the
        <a href="https://www.unifr.ch/home/en/">University of Fribourg</a>, Switzerland. 
        His research interests include <strong>time series analytics</strong> and
        <strong>data quality</strong>, with a focus on <strong>temporal data repair/cleaning</strong>.
        He received the <strong>VLDB 2020 Best Experiments and Analysis Paper Award</strong>.
      </div><br>
      👉 <a href="https://mkhayati.github.io/">Home Page</a>
    </td>
  </tr>
</table>


<br> <br>

---


## References

<a name="ref1"></a>
[1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

<a name="ref2"></a>
[2] Olga G. Troyanskaya, Michael N. Cantor, Gavin Sherlock, Patrick O. Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman: Missing value estimation methods for DNA microarrays. Bioinform. 17(6): 520-525 (2001)

<a name="ref3"></a>
[3] Laura Balzano, Robert D. Nowak, and Benjamin Recht. "Online Identification and Tracking of Subspaces from Highly Incomplete Information." CoRR abs/1006.4046, 2010.

<a name="ref4"></a>
[4] Xianbiao Shu, Fatih Porikli, Narendra Ahuja: Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-Rank Matrices. CVPR 2014: 3874-3881

<a name="ref5"></a>
[5] Spiros Papadimitriou, Jimeng Sun, Christos Faloutsos: Streaming Pattern Discovery in Multiple Time-Series. VLDB 2005: 697-708

<a name="ref6"></a>
[6] Rahul Mazumder, Trevor Hastie, Robert Tibshirani: Spectral Regularization Algorithms for Learning Large Incomplete Matrices. J. Mach. Learn. Res. 11: 2287-2322 (2010)

<a name="ref7"></a>
[7] Jian-Feng Cai, Emmanuel J. Candès, Zuowei Shen: A Singular Value Thresholding Algorithm for Matrix Completion. SIAM J. Optim. 20(4): 1956-1982 (2010)

<a name="ref8"></a>
[8] Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon: Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction. NeurIPS 2016: 847-855

<a name="ref9"></a>
[9] Xiuwen Yi, Yu Zheng, Junbo Zhang, Tianrui Li: ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data. IJCAI 2016: 2704-2710

<a name="ref10"></a>
[10] Lei Li, James McCann, Nancy S. Pollard, Christos Faloutsos: DynaMMo: mining and summarization of coevolving sequences with missing values. 507-516

<a name="ref11"></a>
[11] Kevin Wellenzohn, Michael H. Böhlen, Anton Dignös, Johann Gamper, Hannes Mitterer: Continuous Imputation of Missing Values in Streams of Pattern-Determining Time Series. EDBT 2017: 330-341

<a name="ref12"></a>
[12] Aoqian Zhang, Shaoxu Song, Yu Sun, Jianmin Wang: Learning Individual Models for Imputation. ICDE (2019)

<a name="ref13"></a>
[13] Tianqi Chen, Carlos Guestrin: XGBoost: A Scalable Tree Boosting System. KDD 2016: 785-794

<a name="ref14"></a>
[14] Royston Patrick , White Ian R.: Multiple Imputation by Chained Equations (MICE): Implementation in Stata. Journal of Statistical Software 2011: 45(4), 1–20.

<a name="ref15"></a>
[15] Daniel J. Stekhoven, Peter Bühlmann: MissForest - non-parametric missing value imputation for mixed-type data. Bioinform. 28(1): 112-118 (2012)

<a name="ref22"></a>
[22] Jinsung Yoon, William R. Zame, Mihaela van der Schaar: Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks. IEEE Trans. Biomed. Eng. 66(5): 1477-1490 (2019)

<a name="ref23"></a>
[23] Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, Yitan Li: BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018: 6776-6786

<a name="ref24"></a>
[24] Parikshit Bansal, Prathamesh Deshpande, Sunita Sarawagi: Missing Value Imputation on Multidimensional Time Series. Proc. VLDB Endow. 14(11): 2533-2545 (2021)

<a name="ref25"></a>
[25] Xiao Li, Huan Li, Hua Lu, Christian S. Jensen, Varun Pandey, Volker Markl: Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (Extended Version). Proceedings of the VLDB Endowment, 17(3):345–358, 2023.

<a name="ref26"></a>
[26]: Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu: PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. ICDE 2023: 1927-1939

<a name="ref27"></a>
[27] Kohei Obata, Koki Kawabata, Yasuko Matsubara, Yasushi Sakurai: Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series. KDD 2024: 2296-2306

<a name="ref28"></a>
[28] Jinsung Yoon, James Jordon, Mihaela van der Schaar: GAIN: Missing Data Imputation using Generative Adversarial Nets. ICML 2018: 5675-5684

<a name="ref29"></a>
[29] Andrea Cini, Ivan Marisca, and Cesare Alippi. "Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks." The Tenth International Conference on Learning Representations (ICLR), 2022.

<a name="ref30"></a>
[30] Shikai Fang, Qingsong Wen, Yingtao Luo, Shandian Zhe, Liang Sun: BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition. ICML 2024 (PMLR 235): 12993-13009.

<a name="ref31"></a>
[31] Liang Wang, Simeng Wu, Tianheng Wu, Xianping Tao, Jian Lu: HKMF-T: Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization. IEEE Trans. Knowl. Data Eng. 33(11): 3582-3593 (2021)

<a name="ref32"></a>
[32] Xiaodan Chen, Xiucheng Li, Bo Liu, Zhijun Li: Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values. ICLR 2024

<a name="ref33"></a>
[33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)

<a name="ref34"></a>
[34] Mourad Khayati, Quentin Nater, Jacques Pasquier: ImputeVIS: An Interactive Evaluator to Benchmark Imputation Techniques for Time Series Data. Proc. VLDB Endow. 17(12): 4329-4332 (2024)

<a name="ref35"></a>
[35] Jinguo Cheng, Chunwei Yang, Wanlin Cai, Yuxuan Liang, Qingsong Wen, Yuankai Wu: NuwaTS: a Foundation Model Mending Every Incomplete Time Series. Arxiv 2024

<a name="ref36"></a>
[36] Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin: One fits all: power general time series analysis by pretrained LM. NeurIPS 2023

<a name="ref37"></a>
[37] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023

<a name="ref38"></a>
[38] Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon: CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS 2021



<a name="ref39"></a>
[39] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski: MOMENT: A Family of Open Time-series Foundation Models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 (Proceedings of Machine Learning Research). PMLR OpenReview.net, 16115–16152.

<a name="ref40"></a>
[40] Xiaoye Miao, Yangyang Wu, Jun Wang, Yunjun Gao, Xudong Mao, Jianwei Yin : Generative Semi-supervised Learning for Multivariate Time Series Imputation. AAAI 2021

<a name="ref41"></a>
[41] Wenjie Du, David Côté, Yan Liu : SAITS: : Self-attention-based imputation for time series . ESWA 2023

<a name="ref42"></a>
[42] Vincent Fortuin, Dmitry Baranchuk, Gunnar Rätsch,  Stephan Mandt : GP-VAE: Deep Probabilistic Multivariate Time Series Imputation. AISTATS 2020

<a name="ref43"></a>
[43] Yukai Liu, Rose Yu, Stephan Zheng, Eric Zhan, Yisong Yue : NAOMI: Non-Autoregressive Multiresolution Sequence Imputation. NeurIPS 2019

<a name="ref44"></a>
[44] Shanghua Gao, Teddy Koker, Owen Queen, Thomas Hartvigsen, Theodoros Tsiligkaridis, Marinka Zitnik : UniTS: A Unified Multi-Task Time Series Model. NeurIPS 2024

<a name="ref45"></a>
[45] Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, Mingsheng Long : Timer: Generative Pre-trained Transformers Are Large Time Series Models. ICML 2024



---

<br> <br>

🚧 = under integration