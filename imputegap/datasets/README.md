<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# ImputeGAP - Datasets
ImputeGap brings a repository of highly curated time series datasets for missing values imputation. Those datasets contain
real-world time series from various of applications and which cover a wide range of characteristics and sizes. 


## Air-Quality

The air quality dataset includes a subset of air quality measurements collected from 36 monitoring stations in China from 2014 to 2015.

### Summary

| Data info   |                                                                                                                                                   |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| codename    | airq                                                                                                                                              |
| name        | Air Quality                                                                                                                                       |
| url         | https://archive.ics.uci.edu/dataset/360/air+quality                                                                                               | 
| source      | On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario: Sensors and Actuators B: Chemical'08 | 
| granularity | hourly                                                                                                                                            |
| series      | n_series=10; n_values/series=1000                                                                                                                 |


### Sample plots

![AIR-QUALITY dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/airq/01_airq_m.jpg)


<br /><hr /><br />



## BAFU

The BAFU dataset, kindly provided by the BundesAmt Für Umwelt (the Swiss Federal Office for the Environment)[https://www.bafu.admin.ch], contains water discharge time series collected from different Swiss rivers containing between 200k and 1.3 million values each and covers the time period from 1974 to 2015.

### Summary

| Data info   |                                                                                                               |
|-------------|---------------------------------------------------------------------------------------------------------------|
| codename    | bafu                                                                                                          |
| name        | Water Quality                                                                                                 |
| url         | https://github.com/eXascaleInfolab/bench-vldb20/tree/master                                                   |
| source      | Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series (PVLDB'20) |
| granularity | 30 minutes                                                                                                    |
| size        | n_series=12; n_values/series=85203                                                                            |



### Sample Plots

![BAFU dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/bafu/01_bafu_m.jpg)

<br /><hr /><br />


## Chlorine

The Chlorine dataset originates from chlorine residual management aimed at ensuring the security of water distribution systems [Chlorine Residual Management for Water Distribution System Security](https://www.researchgate.net/publication/226930242_Chlorine_Residual_Management_for_Water_Distribution_System_Security), with data sourced from [US EPA Research](https://www.epa.gov/research).
It consists of 50 time series, each representing a distinct location, with 1,000 data points per series recorded at 5-minute intervals.


### Summary

| Data info            |                                                                |
|-------------------------|----------------------------------------------------------------|
| codename                | chlorine                                                       |
| name                    | Chlorine data                                                  |
| URL                     | https://www.epa.gov/research                                   |
| source                  | Streaming pattern discovery in multiple time-series (PVLDB'05) |
| granularity             | 5 minutes                                                      |
| size                    | n_series=50; n_values/series=1000                              |


### Sample Plots

![Chlorine dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/chlorine/01_chlorine_m.jpg)


<br /><hr /><br />




## Climate

The Climate dataset is an aggregated and processed collection used for climate change attribution studies.
It contains observations data for 18 climate agents across 125 locations in North America [USC Melady Lab](https://viterbi-web.usc.edu/~liu32/data.html).
The dataset has a temporal granularity of 1 month, comprising 10 series with 5,000 values each.
This structure is particularly valuable for spatio-temporal modeling [Spatial-temporal causal modeling for climate change attribution](https://dl.acm.org/doi/10.1145/1557019.1557086), as it enables researchers to account for both spatial and temporal dependencies.

### Summary

| Data info   |                                                                          |
|-------------|--------------------------------------------------------------------------|
| codename    | climate                                                                  |
| name        | Aggregated and Processed data collection for climate change attribution  |
| url         | https://viterbi-web.usc.edu/~liu32/data.html (NA-1990-2002-Monthly.csv)  |
| source      | Spatial-temporal causal modeling for climate change attribution (KDD'09) |
| granularity | 1 month                                                                  |
| size        | n_series=10; n_values/series=5000                                        |



### Sample Plots

![Climate dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/climate/01_climate_m.jpg)


<br /><hr /><br />


## Drift
The Drift dataset comprises 13,910 measurements collected from 16 chemical sensors exposed to six different gases, with only batch 10 utilized for this dataset [Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270).
It includes information on the concentration levels to which the sensors were exposed during each measurement.
Data was collected over a 36-month period, from January 2008 to February 2011, at a gas delivery platform facility within the ChemoSignals Laboratory at the BioCircuits Institute, University of California, San Diego [On the calibration of sensor arrays for pattern recognition using the minimal number of experiments](https://www.sciencedirect.com/science/article/pii/S0169743913001937).
The dataset has a time granularity of 6 hours and consists of 100 time series, each containing 1,000 data points. 

### Summary

| Data info   |                                                                                                                                                          |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| codename    | drift                                                                                                                                                    |
| name        | Gas Sensor Array Drift Dataset at Different Concentrations                                                                                               |
| url         | https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations (only batch 10)                                       |
| source      | On the calibration of sensor arrays for pattern recognition using the minimal number of experiments (Chemometrics and Intelligent Laboratory Systems'14) |
| granularity | 6 hours                                                                                                                                                  |
| size        | n_series=100; n_values/series=1000                                                                                                                       |


### Sample Plots


![Drift dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/drift/01_drift_m.jpg)

<br /><hr /><br />



## EEG-Alcohol

The EEG-Alcohol dataset, owned by Henri Begleiter [EEG dataset](https://kdd.ics.uci.edu/databases/eeg/eeg.data.html), is utilized in various studies such as [Statistical mechanics of neocortical interactions: Canonical momenta indicatorsof electroencephalography](https://link.aps.org/doi/10.1103/PhysRevE.55.4578).
It describes an EEG database composed of individuals with a genetic predisposition to alcoholism.
The dataset contains measurements from 64 electrodes placed on subject's scalps which were sampled at 256 Hz (3.9-msec epoch) for 1 second.
The dataset contains a total of 416 samples.
The specific subset used in ImputeGAP is the S2 match for trial 119, identified as `co3a0000458.rd`.
The dataset's dimensions are 64 series, each containing 256 values.
This dataset is primarily used for the analysis of medical and brain-related data, with a focus on detecting predictable patterns in brain wave activity.


### Summary

| Data info   | Values                                                                                                                                                              |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codename    | eeg-alcohol                                                                                                                                                         |
| name        | EEG Database: Genetic Predisposition to Alcoholism                                                                                                                  |
| url         | https://kdd.ics.uci.edu/databases/eeg/eeg.data.html (batch co3a0000458.rd / S2 match, trial 119)                                                                    |
| source      | Statistical mechanics of neocortical interactions: Canonical momenta indicators of electroencephalography. Physical Review E. Volume 55. Number 4. Pages 4578-4593. |
| granularity | 1 second per measurement (3.9 ms epoch)                                                                                                                             |
| size        | n_series=64; n_values/series=256                                                                                                                                    |



### Sample Plots

![EEG-ALCOHOL dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/eeg-alcohol/01_eeg_alcohol_m.jpg)


<br /><hr /><br />





## EEG-READING

The EEG-Reading dataset, created by the DERCo, is a collection of EEG recordings obtained from participants engaged in text reading tasks [A Dataset for Human Behaviour in Reading Comprehension Using {EEG}](https://www.nature.com/articles/s41597-024-03915-8).
This corpus includes behavioral data from 500 participants, as well as EEG recordings from 22 healthy adult native English speakers.
The dataset features a time resolution of 1000 Hz, with time-locked recordings from -200 ms to 1000 ms relative to the stimulus onset.
The dataset consists of 564 epochs, although only one was selected for this specific EEG subset.
The extracted dataset contains 1201 values across 33 series.


### Summary

| Data info   |                                                                                              |
|-------------|----------------------------------------------------------------------------------------------|
| codename    | eeg-reading                                                                                  |
| name        | DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG                      |
| url         | https://doi.org/10.17605/OSF.IO/RKQBU (epoch 1 used on 564)                                  |
| source      | DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG (scientific data'24) |
| granularity | 1000.0 Hz                                                                                    |
| size        | n_series=33; n_values/series=1202                                                            |



### Sample Plots

![EEG-READING dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/eeg-reading/01_eeg_reading_m.jpg)





<br /><hr /><br />






## Electricity

The electricity dataset has data on household energy consumption of 370 individual clients collected every minute between 2006 and 2010 in France (obtained from the UCI repository.


### Summary

| Data info   |                                                                                          |
|-------------|------------------------------------------------------------------------------------------|
| codename    | electricity                                                                              |
| name        | Household Electricity Consumption                                                        |
| url         | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014                  | 
| source      | Artur Trindade, artur.trindade '@' elergone.pt <br> Elergone, NORTE-07-0202-FEDER-038564 | 
| granularity | 15 minutes                                                                               |
| size        | n_series=20; n_values/series=5000                                                        |



### Sample Plots

![ELECTRICITY dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/electricity/01_electricity_M.jpg)





<br /><hr /><br />



## fMRI-Stoptask

The fMRI-Stoptask dataset was obtained from the OpenfMRI database, with the accession number ds000007. This dataset is an extraction of a fMRI scan of Visual where subjects performed a stop-signal task with one of three response types: manual response, spoken letter naming, and spoken pseudo word naming.
Following the same conversion hypothesis as used for the object recognition dataset, the fMRI-Stoptask dataset was extracted from the first run of subject 1. Voxels with values of 0 were removed, and the total number of voxels was reduced to 10,000 after flattening the dimensions. This resulted in a dataset comprising 10,000 series, each containing 182 values.
The fMRI-Stoptask dataset will emphasize brain activity in regions such as the right inferior frontal gyrus and the basal ganglia, illustrating neural mechanisms of inhibition commonly associated with stop-signal tasks.


### Summary

| Data  info |                                                                                           |
|------------|-------------------------------------------------------------------------------------------|
| codename   | fmri-objectviewing                                                                        |
| name       | Visual object recognition                                                                 |
| url        | https://www.openfmri.org/dataset/ds000007/ (epoch 1 used on 120)                          |
| source     | Common neural substrates for inhibition of spoken and manual responses. (Cereb Cortex'08) |
| size       | n_series=10000; n_values/series=182                                                       |




### Sample Plots

![fMRI-STOPTASK](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/fmri-stoptask/01_fmri_stoptask_m.jpg)

<br /><hr /><br />




## MeteoSwiss

The MeteoSwiss dataset, kindly provided by the Swiss Federal Office of Meteorology and Climatology [http://meteoswiss.admin.ch], contains weather time series recorded in different cities in Switzerland from 1980 to 2018. The MeteoSwiss dataset appeared in [[1]](#ref1).

### Summary

| Data  info  |                                                                                                                                |
|-------------|--------------------------------------------------------------------------------------------------------------------------------|
| codename    | meteo                                                                                                                          |
| name        | Meteo Suisse data                                                                                                              |
| url         | https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html (meteo_total_08-12.txt with 9999 first lines) | 
| source      | Scalable Recovery of Missing Blocks in Time Series with High and Low Cross-Correlations (KAIS'20)                              | 
| granularity | 10 minutes                                                                                                                     |
| size        | n_series=4; n_values/series=9999                                                                                               |

### Sample Plots

![Meteo dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/meteo/01_meteo_m.jpg)


<br /><hr /><br />




## Motion

The motion dataset consists of time series data collected from accelerometer and gyroscope sensors, capturing attributes such as attitude, gravity, user acceleration, and rotation rate [[4]](#ref4). Recorded at a high sampling rate of 50Hz using an iPhone 6s placed in users' front pockets, the data reflects various human activities. While the motion time series are non-periodic, they display partial trend similarities.

### Summary

| Data info   |                                              |
|-------------|----------------------------------------------|
| codename    | motion                                       |
| name        | Motion Sense                                 |
| url         | https://github.com/mmalekzadeh/motion-sense  | 
| source      | Mobile Sensor Data Anonymization (IoTDI ’19) | 
| granularity | 50Hz                                         |
| size        | n_series=20; n_values/series=10000           |


### Sample Plots

![Motion dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/motion/01_motion_M.jpg)




<br /><hr /><br />




## Soccer

The soccer dataset, initially presented in the DEBS Challenge 2013 [[3]](#ref3), captures player positions during a football match. The data is collected from sensors placed near players' shoes and the goalkeeper's hands. With a high tracking frequency of 200Hz, it generates 15,000 position events per second. Soccer time series exhibit bursty behavior and contain numerous outliers.

### Summary

| Data info |                                         |
|-----------|-----------------------------------------|
| codename  | soccer                                  |
| name      | Soccer                                  |
| url       | https://debs.org/grand-challenges/      |
| source    | The DEBS 2013 grand challenge (DEBS'13) |
| size      | n_series=10; n_values/series=100000     |


### Sample Plots
![Soccer dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/soccer/01_soccer_M.jpg)


<br /><hr /><br />


## Solar

Real-time dataset of a thermal solar plant logged every minute from December 28, 2016, to October 10, 2018. The recordings come directly from a real plant, capturing the following:

- Temperatures from multiple sensors
- Pressure readings
- Flow rates
- PWM and relay activity
- Operational runtime
- Heat energy output
- Error and status codes
- And several other system metrics


### Summary

| Dataset info |                                                                                                 |
|--------------|-------------------------------------------------------------------------------------------------|
| codename     | solar                                                                                           |
| name         | Solar Plant                                                                                     |
| url          | https://github.com/stritti/thermal-solar-plant-dataset (temperature 01/2021 used: 20210120.csv) | 
| source	     | -                                                                                               |
| granularity  | minutes                                                                                         |
| size         | n_series=3; n_values/series=799                                                                 |



### Sample Plots

![SOLAR dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/solar_plant/01_solat_plant_M.jpg)



<br /><hr /><br />


## Sport Activity

Dataset consists of data in categories walking, running, biking, skiing, and roller skiing (5). Sport activities have been recorded by an individual active (non-competitive) athlete.

### Summary

| Dataset info |                                                                                         |
|--------------|-----------------------------------------------------------------------------------------|
| codename     | sport_activity                                                                          |
| name         | Sport Activity                                                                          |
| url          | https://www.kaggle.com/datasets/jarnomatarmaa/sportdata-mts-5?resource=download         | 
| source	      | A novel multivariate time series dataset of outdoor sport activities (Discover Data'25) |
| granularity  | 1 minute                                                                                |
| size         | n_series=69; n_values/series=1140                                                       |



### Sample Plots

![SPORT-ACTIVITY dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/sport_activity/01_sport_activity_M.jpg)


<br /><hr /><br />



## Stock Exchange

the collection of the daily exchange rates of eight foreign countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016.


### Summary

| Dataset info |                                                                                                                        |
|--------------|------------------------------------------------------------------------------------------------------------------------|
| codename     | stock_exchange                                                                                                         |
| name         | Stock Exchange                                                                                                         |
| source       | https://github.com/laiguokun/multivariate-time-series-data/tree/master                                                 | 
| paper	       | Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, Arxiv'17 (https://arxiv.org/abs/1703.07015) |
| granularity  | daily                                                                                                                  |
| size         | n_series=8; n_values/series=7588                                                                                       |



### Sample Plots

![STOCK-EXCHANGE dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/stock_exchange/01_stock_exchange_M.jpg)



<br /><hr /><br />



## Temperature

This dataset contains temperature data collected from climate stations in China from 1960 to 2012. Temperature time series are very highly correlated with each other.


### Summary

| Data info   |                                                                           |
|-------------|---------------------------------------------------------------------------|
| codename    | temperature                                                               |
| name        | Temperature                                                               |
| url         | http://www.cma.gov.cn (25 first series)                                   | 
| source      | ST-MVL: filling missing values in geo-sensory time series data (IJCAI'16) |
| granularity | daily                                                                     |
| size        | n_series=25; n_values/series=19358                                        |


### Sample Plots

![Temperature dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/temperature/01_temperature_M.jpg)


<br /><hr /><br />

## Traffic

The raw data is in http://pems.dot.ca.gov. The data in this repo is a collection of 48 months (2015-2016) hourly data from the California Department of Transportation. The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area freeways.


### Summary

| Dataset info |                                                                                      |
|--------------|--------------------------------------------------------------------------------------|
| codename     | traffic                                                                              |
| name         | Traffic                                                                              |
| url          | https://github.com/laiguokun/multivariate-time-series-data/tree/master               | 
| source	      | Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Arxiv'17) |
| granularity  | hourly                                                                               |
| size         | n_series=20; n_values/series=17544                                                   |



### Sample Plots

![TRAFFIC dataset](https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/traffic/01_traffic_M.jpg)


<br /><hr /><br />





## References

<a name="ref1"></a>
[1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

[2] Ines Arous, Mourad Khayati, Philippe Cudré-Mauroux, Ying Zhang, Martin L. Kersten, Svetlin Stalinlov: RecovDB: Accurate and Efficient Missing Blocks Recovery for Large Time Series. ICDE 2019: 1976-1979

[3] Christopher Mutschler, Holger Ziekow, and Zbigniew Jerzak. 2013. The DEBS  2013 grand challenge. In debs, 2013. 289–294

[4] Mohammad Malekzadeh, Richard G. Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile Sensor Data Anonymization. In Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI ’19). ACM,  New York, NY, USA, 49–58. https://doi.org/10.1145/3302505.3310068
