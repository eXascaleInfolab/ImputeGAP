<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# ImputeGAP - Datasets
ImputeGap brings a repository of highly curated time series datasets for missing values imputation. Those datasets contain
real-world time series from various of applications and which cover a wide range of characteristics and sizes. 


## Dataset Index

## 📑 Dataset Index

| Category          | Codename                            | Dataset                                                   |
|-------------------|-------------------------------------|-----------------------------------------------------------|
| **Weather**       | [`climate`](#climate)               | Climate Change                                            |
| **Weather**       | [`meteo`](#meteoswiss)              | MeteoSwiss Weather                                        |
| **Weather**       | [`temperature`](#temperature)       | Temperature                                               |
| **Air Quality**   | [`airq`](#air-quality)              | Air Quality                                               |
| **Air Quality**   | [`drift`](#drift)                   | Gas Sensor Array Drift                                    |
| **Water Quality** | [`bafu`](#bafu)                     | BAFU Water Discharge                                      |
| **Water Quality** | [`chlorine`](#chlorine)             | Chlorine Residual                                         |
| **Neuroscience**  | [`eeg-alcohol`](#eeg-alcohol)       | EEG Alcohol                                               |
| **IoT Sensing**   | [`motion`](#motion)                 | Motion Sensor (Human Activity Recognition)                |
| **IoT Sensing**   | [`traffic`](#traffic)               | PEMS Traffic                                              |
| **Sports**        | [`soccer`](#soccer)                 | Soccer Player Tracking                                    |
| **Sports**        | [`sport_activity`](#sport-activity) | Sport Activity Dataset (Multi-Modal Exercise Recognition) |
| **Energy**        | [`electricity`](#electricity)       | Electricity Consumption                                   |
| **Finance**       | [`stock_exchange`](#stock-exchange) | Exchange Rates                                            |


<br /><br /><hr /><br /><br />


## Air Quality Dataset

The Air Quality dataset contains hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was deployed on the field at road level in a significantly polluted area within an Italian city. Data was recorded from March 2004 to February 2005, representing one year of measurements and the longest freely available recordings of on-field deployed air quality chemical sensor device responses at the time of collection. The sensor array includes five metal oxide sensors (tin oxide, titania, tungsten oxide, indium oxide) nominally targeted at different pollutants, along with measurements of temperature, relative humidity, and absolute humidity.  

Important patterns observable in the data include recurring diurnal and seasonal cycles in pollutant concentrations, traffic- and industry-driven pollution spikes in densely populated areas, weather-induced dispersion or accumulation effects, and regionally synchronized episodes of degraded air quality during adverse meteorological conditions.


<br />


### Details

| Data info |                                                                                                                                                   |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Codename  | airq                                                                                                                                              |
| Source    | https://archive.ics.uci.edu/dataset/360/air+quality                                                                                               | 
| Paper     | On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario: Sensors and Actuators B: Chemical'08 | 
| Frequency | hourly                                                                                                                                            |
| Size      | (1000, 10)                                                                                                                                        |

<br />


### Sample plots

<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/airq/imputegap_airq_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/airq/imputegap_airq_plot.jpg" width="45%" />
</p>




<br /><br /><hr /><br /><br />



## BAFU Water Discharge Dataset

We donate the BAFU Water Discharge dataset, a comprehensive collection of water discharge time series from Swiss river monitoring stations. This dataset comprises measurements from multiple rivers across Switzerland, with each time series containing between 200,000 and 1.3 million data points spanning the period from 1974 to 2015, representing over four decades of continuous hydrological monitoring. The data originates from the BundesAmt Für Umwelt (BAFU), the Swiss Federal Office for the Environment [https://www.bafu.admin.ch], which operates a large monitoring network of automated gauging stations across Swiss watercourses and lakes. This dataset is characterized by a broad geographic diversity across multiple Swiss river systems encompassing different regions, elevations, and climatic zones, captured within a rigorously quality-controlled monitoring framework using standardized protocols.


Notable patterns present in the data include pronounced seasonal flow cycles driven by snowmelt and precipitation, long-term trends associated with climate change, flow alterations linked to hydropower operations, episodic extremes such as floods or droughts, and subtle measurement artifacts. 

<br />


### Details

| Data info   |                                                                                                               |
|-------------|---------------------------------------------------------------------------------------------------------------|
| Codename    | bafu                                                                                                          |
| Source      | https://github.com/eXascaleInfolab/bench-vldb20/tree/master                                                   |
| Paper       | Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series (PVLDB'20) |
| Frequency   | 30 minutes                                                                                                    |
| Size        | (85203, 12)                                                                                                   |

<br />


### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/bafu/imputegap_bafu_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/bafu/imputegap_bafu_1_plot.jpg" width="45%" />
</p>




<br /><br /><hr /><br /><br />



## Chlorine Residual Dataset

The Chlorine Residual dataset was generated by EPANET 2.0, which accurately simulates hydraulic and chemical phenomena within drinking water distribution systems. The dataset monitors chlorine concentration at all 166 junctions in a water distribution network for 4,310 timestamps over 15 days, with measurements recorded every five minutes. Chlorine concentration patterns in the network arise primarily from water demand dynamics. When water is not refreshed in pipes, chlorine reacts with pipe walls and microorganisms, causing concentration to drop. Fresh water flow due to demand causes concentrations to rise again, with levels depending mainly on the chlorine concentration originally mixed at reservoirs and, to a lesser extent, the distance to the nearest reservoir (as concentrations drop slightly due to reactions along the way).

The dataset exhibits two key characteristics: a clear global periodic pattern reflecting the daily demand cycle, where chlorine concentrations follow a near-sinusoidal pattern, and slight temporal phase shifts across different junction locations due to the time required for fresh water to flow through the pipes from the reservoirs.

<br />


### Details

| Data info |                                                                |
|-----------|----------------------------------------------------------------|
| Codename  | chlorine                                                       |
| Source    | https://www.epa.gov/research                                   |
| Paper     | Streaming pattern discovery in multiple time-series (PVLDB'05) |
| Frequency | 5 minutes                                                      |
| Size      | (1000, 50)                                                     |

<br />


### Sample Plots

<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/chlorine/imputegap_chlorine_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/chlorine/imputegap_chlorine_plot.jpg" width="45%" />
</p>




<br /><br /><hr /><br /><br />




## Climate Change Dataset

The Climate dataset is a curated collection designed specifically for climate change attribution studies. The data originates from the USC Melady Lab and aggregates observations of 18 distinct climate variables across 125 monitoring locations distributed throughout North America. The dataset captures monthly measurements spanning multiple decades, providing the temporal scale necessary to identify long-term climate trends, seasonal cycles, and inter-annual variability. The 18 climate variables typically include temperature (surface and atmospheric), precipitation, humidity, pressure, wind patterns, solar radiation, and other atmospheric and oceanic indicators relevant to climate system dynamics.

This dataset serves as a foundation for understanding the complex causal relationships between different climate agents and their spatiotemporal patterns. It has continental-scale spatial coverage and a spatiotemporal structure explicitly designed for modeling spatial dependencies (geographic proximity effects) and temporal dependencies (lagged relationships, seasonal patterns).

<br />


### Details

| Data info |                                                                          |
|-----------|--------------------------------------------------------------------------|
| Codename  | climate                                                                  |
| Source    | https://viterbi-web.usc.edu/~liu32/data.html (NA-1990-2002-Monthly.csv)  |
| Paper     | Spatial-temporal causal modeling for climate change attribution (KDD'09) |
| Frequency | 1 month                                                                  |
| Size      | (5000, 10)                                                               |

<br />


### Sample Plots

<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/climate/imputegap_climate_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/climate/imputegap_climate_plot.jpg" width="45%" />
</p>




<br /><br /><hr /><br /><br />



## Gas Sensor Array Drift Dataset
The Gas Sensor Array Drift at Different Concentrations dataset contains measurements from 16 chemical sensors exposed to 6 different gases at various concentration levels, collected at the ChemoSignals Laboratory in the BioCircuits Institute, University of California San Diego. Data collection spanned 36 months from January 2008 to February 2011. The dataset includes recordings from six pure gaseous substances: Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, and Toluene, dosed at different concentration levels. Each measurement produced a 16-channel time series, with 8 features extracted from each sensor, resulting in a 128-dimensional feature vector.

Key patterns in the dataset include temporal drift where sensor responses change over time due to aging, poisoning, and environmental factors; progressive drift evolution organized across ten temporal batches; and multi-gas discrimination challenges across six distinct chemical substances with overlapping concentration ranges.


<br />


### Details

| Data info |                                                                                                                                                          |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Codename  | drift                                                                                                                                                    |
| Source    | https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations (only batch 10)                                       |
| Paper     | On the calibration of sensor arrays for pattern recognition using the minimal number of experiments (Chemometrics and Intelligent Laboratory Systems'14) |
| Frequency | 6 hours                                                                                                                                                  |
| Size      | (1000, 100)                                                                                                                                              |

<br />


### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/drift/imputegap_drift_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/drift/imputegap_drift_plot.jpg" width="45%" />
</p>



<br /><br /><hr /><br /><br />



## EEG-Alcohol Dataset

The EEG-Alcohol dataset is a neurophysiological recording collection originally compiled by Henri Begleiter for studying genetic predisposition to alcoholism. The dataset contains high-resolution brain electrical activity recordings from individuals selected based on their genetic risk factors for alcohol use disorders. Each recording captures synchronized measurements from 64 electrodes positioned according to the international 10-20 system (or extended variants) across the subject's scalp, providing comprehensive spatial coverage of cortical activity. The recordings were acquired at a sampling rate of 256 Hz (3.9-millisecond temporal resolution) over 1-second epochs, yielding fine-grained temporal dynamics of neural oscillations. The complete database comprises 416 samples collected under controlled experimental conditions, typically involving cognitive tasks or stimulus-response paradigms designed to elicit measurable differences in brain activity patterns between individuals with and without genetic predisposition to alcoholism. For the ImputeGAP library, we utilize a specific subset: the S2 match condition for trial 119, identified by the recording file `co3a0000458.rd`.

The dataset has a high spatial resolution with a 64-channel electrode array capturing distributed brain activity across multiple cortical regions, high temporal resolution at 256 Hz enabling analysis of fast neural oscillations including gamma band activity, short-duration, high-density structure with 256 values per channel ideal for testing algorithms on brief but information-rich time series, and complex signals exhibiting multi-scale temporal dynamics, and non-stationarity.

<br />


### Details

| Data info | Values                                                                                                                                                              |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Codename  | eeg-alcohol                                                                                                                                                         |
| Source    | https://kdd.ics.uci.edu/databases/eeg/eeg.data.html (batch co3a0000458.rd / S2 match, trial 119)                                                                    |
| Paper     | Statistical mechanics of neocortical interactions: Canonical momenta indicators of electroencephalography. Physical Review E. Volume 55. Number 4. Pages 4578-4593. |
| Frequency | 1 second per measurement (3.9 ms epoch)                                                                                                                             |
| Size      | (256, 64)                                                                                                                                                           |

<br />


### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/eeg-alcohol/imputegap_eeg-alcohol_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/eeg-alcohol/imputegap_eeg-alcohol_plot.jpg" width="45%" />
</p>


<br /><br /><hr /><br /><br />




## Electricity Consumption Dataset

The Electricity dataset is a large-scale collection of electricity consumption measurements from individual clients in Portugal. The data originates from the UCI Machine Learning Repository and represents one of the most comprehensive publicly available datasets for studying fine-grained electricity usage patterns. The dataset contains electricity consumption readings from 370 individual clients monitored continuously over a period from 2011 to 2014. This extended monitoring period captures seasonal variations, yearly trends, and evolving consumption behaviors across clients with diverse consumption patterns and energy usage habits. Each client's consumption is recorded as power in kilowatts at 15-minute intervals (96 measurements per day), providing high temporal granularity for energy analysis.

Key patterns include intra-day fluctuations and multi-scale periodicity with daily, weekly, and seasonal cycles. This dataset is valuable for time series analysis as it provides real-world complexities including multiple periodicities, temporal irregularities, and a large ensemble of correlated series.

<br />


### Details

| Data info   |                                                                                          |
|-------------|------------------------------------------------------------------------------------------|
| Codename    | electricity                                                                              |
| Source      | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014                  | 
| Paper       | Artur Trindade, artur.trindade '@' elergone.pt <br> Elergone, NORTE-07-0202-FEDER-038564 | 
| Frequency   | 15 minutes                                                                               |
| Size        | (5000, 20)                                                                               |

<br />


### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/electricity/imputegap_electricity_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/electricity/imputegap_electricity_plot.jpg" width="45%" />
</p>



<br /><br /><hr /><br /><br />



## MeteoSwiss Weather Dataset

We donate the MeteoSwiss dataset, a collection of meteorological time series from Switzerland's national weather monitoring network operated by MeteoSwiss (http://meteoswiss.admin.ch). This dataset spans nearly four decades of high-quality observations suitable for long-term trend analysis, seasonal decomposition, and climate pattern modeling. The dataset contains measurements from multiple monitoring stations across 1980 to 2018 (38 years), creating a multivariate spatial-temporal structure with synchronized observations across different geographic locations. The automated stations measure multiple meteorological variables (temperature, precipitation, humidity, atmospheric pressure, wind speed, wind direction, sunshine duration), forming co-evolving time series with complex interdependencies.

Key time series properties include: strong seasonal periodicity with annual cycles, nested seasonality combining daily and yearly patterns, non-stationarity with climate warming trends, spatial heterogeneity across elevation zones and climate regions, extreme value distributions in precipitation and temperature. This enables analysis of spatial correlations, regional co-movements, and elevation-dependent patterns.

<br />


### Details

| Data  info |                                                                                                                                |
|------------|--------------------------------------------------------------------------------------------------------------------------------|
| Codename   | meteo                                                                                                                          |
| Source     | https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html (meteo_total_08-12.txt with 9999 first lines) | 
| Paper      | Scalable Recovery of Missing Blocks in Time Series with High and Low Cross-Correlations (KAIS'20)                              | 
| Frequency  | 10 minutes                                                                                                                     |
| Size       | (9999, 4)                                                                                                                      |

<br />


### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/meteo/imputegap_meteo_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/meteo/imputegap_meteo_plot.jpg" width="45%" />
</p>



<br /><br /><hr /><br /><br />





## Motion Sensor Dataset

The Motion dataset is a collection of inertial sensor measurements capturing human movement patterns during various daily activities. The data originates from smartphone-embedded motion sensors and represents high-frequency recordings of body dynamics during natural locomotion and activities, as utilized in research on human activity recognition [[4]](#ref4). The dataset comprises synchronized time series from multiple inertial measurement unit (IMU) sensors integrated within an iPhone 6s smartphone. Specifically, the recordings include data from both accelerometer and gyroscope sensors, yielding multiple complementary motion attributes (Attitude, Gravity vector, User acceleration, Rotation rate). All measurements were recorded at a sampling frequency of 50 Hz, providing 20-millisecond temporal resolution sufficient to capture the dynamics of human movement including walking cadence, gesture transitions, and rapid motion changes. 

The dataset exhibits non-periodic temporal structure with partial trend similarities, meaning that while activities don't repeat in strictly regular cycles, certain motion patterns recur (e.g., stride phases during walking) and different activities may share common movement elements (e.g., vertical oscillations). This characteristic makes the dataset challenging and realistic for algorithm evaluation, as methods must handle irregular temporal patterns without relying on simple periodicity assumptions.

<br />


### Details

| Data info  |                                              |
|------------|----------------------------------------------|
| Codename   | motion                                       |
| Source     | https://github.com/mmalekzadeh/motion-sense  | 
| Paper      | Mobile Sensor Data Anonymization (IoTDI ’19) | 
| Frequency  | 50Hz                                         |
| Size       | (10000, 20)                                  |


<br />


### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/motion/imputegap_motion_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/motion/imputegap_motion_plot.jpg" width="45%" />
</p>


<br /><br /><hr /><br /><br />




## Soccer Player Tracking Dataset 

The Soccer dataset is a high-resolution collection of real-time player position tracking data from professional football matches. The dataset was originally introduced as part of the DEBS (Distributed Event-Based Systems) Grand Challenge 2013 [[3]](#ref3), which focused on real-time complex event processing and analytics for sports applications. The data is collected using a sophisticated sensor-based tracking system where miniaturized wireless sensors are strategically positioned on each player's body, specifically mounted near both shoes for field players and on the goalkeeper's hands to capture their unique movement patterns and ball-handling actions. These sensors continuously broadcast their spatial coordinates, enabling reconstruction of player trajectories, movement patterns, tactical formations, and game dynamics with unprecedented temporal precision. The tracking system operates at an exceptionally high sampling frequency of 200 Hz, generating position updates every 5 milliseconds per sensor. With multiple players (typically 20+ field players plus goalkeepers) tracked simultaneously, the system produces approximately 15,000 position events per second across the entire match, resulting in massive high-velocity data streams. 

The dataset has bursty nature. The position changes are highly non-uniform, players may remain relatively stationary during ball possession far from their position, then exhibit explosive accelerations when chasing the ball or making attacking runs. This temporal heterogeneity challenges algorithms designed for smooth, regularly-varying time series. The numerous outliers arise from sensor limitations (GPS/RF multipath, body occlusion), physical contact between players causing sensor displacement, and legitimate extreme movements (sliding tackles, collisions, rapid pivots) that appear anomalous but represent genuine athletic actions.

<br />


### Details

| Data info |                                         |
|-----------|-----------------------------------------|
| Codename  | soccer                                  |
| Source    | https://debs.org/grand-challenges/      |
| Paper     | The DEBS 2013 grand challenge (DEBS'13) |
| Size      | (100000, 10)                            |


<br />



### Sample Plots

<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/soccer/imputegap_soccer_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/soccer/imputegap_soccer_plot.jpg" width="45%" />
</p>




<br /><br /><hr /><br /><br />



## Sport Activity Dataset

The Sport Activity dataset is a collection of inertial sensor measurements capturing five distinct athletic activities performed by an individual active, non-competitive athlete. The dataset includes recordings from walking, running, biking, skiing, and roller skiing, representing a diverse range of locomotion modes spanning both land-based and snow-based activities with varying biomechanical characteristics and intensity levels.
The athlete performed each activity naturally, following typical training or recreational exercise patterns, which means the data captures genuine movement dynamics, fatigue effects, terrain variations, and technique variations that occur during actual athletic practice. 

The dataset contains multi-modal activity diversity with abrupt transitions that indicate either a change in activity type, a substantial increase in movement intensity, or the onset of a high-exertion phase. The moderate variability throughout the earlier timestamps, punctuated by occasional spikes, demonstrates the natural variation inherent in human athletic performance, including changes in pace, technique adjustments, and environmental factors.

<br />


### Details

| Dataset info |                                                                                         |
|--------------|-----------------------------------------------------------------------------------------|
| Codename     | sport-activity                                                                          |
| Source       | https://www.kaggle.com/datasets/jarnomatarmaa/sportdata-mts-5?resource=download         | 
| Paper        | A novel multivariate time series dataset of outdoor sport activities (Discover Data'25) |
| Frequency    | 1 minute                                                                                |
| Size         | (1140, 69)                                                                              |


<br />

### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/sport-activity/imputegap_sport-activity_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/sport-activity/imputegap_sport-activity_plot.jpg" width="45%" />
</p>




<br /><br /><hr /><br /><br />



## Stock Exchange Rates Dataset

The Exchange Rate dataset is a collection of daily exchange rates for eight foreign currencies spanning over two decades. The dataset contains daily observations from 1990 to 2016, providing a long temporal span of 26 years with consistent daily sampling frequency suitable for long-term trend analysis, seasonal pattern detection, and regime change identification across multiple economic cycles. The dataset includes exchange rates for eight countries: Australia, British (United Kingdom), Canada, Switzerland, China, Japan, New Zealand, and Singapore, creating a multivariate time series structure. All exchange rates are quoted against a common base currency, enabling synchronized multi-series analysis, cross-correlation studies, and cointegration testing across different economic zones.

Key properties of the dataset include: volatility clustering exhibiting temporal dependence in variance (ARCH/GARCH effects), mean reversion tendencies suggesting stationarity around long-term equilibrium levels, structural breaks and regime changes marking distinct statistical periods (1997 Asian crisis, 2008 financial crisis), and cross-series correlations revealing lead-lag relationships and common stochastic trends, particularly between regional neighbors (AUD-NZD) or asset classes (commodity currencies: AUD, CAD, NZD).

<br />


### Details

| Dataset info |                                                                                                                        |
|--------------|------------------------------------------------------------------------------------------------------------------------|
| Codename     | stock-exchange                                                                                                         |
| Source       | https://github.com/laiguokun/multivariate-time-series-data/tree/master                                                 | 
| Paper        | Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, Arxiv'17 (https://arxiv.org/abs/1703.07015) |
| Frequency    | daily                                                                                                                  |
| Size         | (7588, 8)                                                                                                              |


<br />

### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/stock-exchang/imputegap_stock-exchange_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/stock-exchang/imputegap_stock-exchange_plot.jpg" width="45%" />
</p>



<br /><br /><hr /><br /><br />


## China Climate Temperature Dataset

The Temperature dataset is a long-term collection of surface air temperature measurements from meteorological stations distributed across China. The data spans 52 years from 1960 to 2012, capturing over five decades of temperature observations during a period of significant climate change, rapid industrialization, and urbanization in China. This dataset represents one of the longest continuous climate monitoring records available from a major developing nation experiencing substantial economic and environmental transformation. The high correlation between temperature time series is a distinctive characteristic of this dataset, temperature is a spatially smooth field where nearby locations experience similar atmospheric conditions, weather systems, and seasonal forcing. Correlations typically decay with distance but remain substantial across hundreds of kilometers, particularly for stations in the same climate zone or influenced by common synoptic patterns (e.g., monsoon systems, Siberian high-pressure influences). 

Notable visible patterns include: strong seasonal cycles with warm summers and cold winters (particularly pronounced in northern continental stations), long-term warming trends (especially strong after 1980s, consistent with global climate change), urban heat island signatures in major cities (Beijing, Shanghai, Guangzhou showing accelerated warming), monsoon-related precipitation-temperature relationships in southern and eastern regions, and inter-annual variability.

<br />

### Details

| Dataset info |                                                                           |
|--------------|---------------------------------------------------------------------------|
| Codename     | temperature                                                               |
| Source       | http://www.cma.gov.cn (25 first series)                                   | 
| Paper        | ST-MVL: filling missing values in geo-sensory time series data (IJCAI'16) |
| Frequency    | daily                                                                     |
| Size         | (19358, 25)                                                               |


<br />

### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/temperature/imputegap_temperature_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/temperature/imputegap_temperature_plot.jpg" width="45%" />
</p>



<br /><br /><hr /><br /><br />



## PEMS Traffic Dataset

The PEMS Traffic dataset is a collection of freeway traffic measurements from California's Performance Measurement System (PeMS), available at http://pems.dot.ca.gov. The dataset contains 48 months of hourly observations from 2015 to 2016, providing consistent temporal sampling suitable for short-term forecasting and seasonality analysis across the San Francisco Bay Area freeway network.
The dataset focuses on road occupancy rates (values between 0 and 1), exhibiting strong diurnal periodicity with rush hour peaks, day-of-week seasonality distinguishing weekday from weekend patterns, and trend components reflecting long-term traffic demand changes. Measurements are collected from multiple sensor stations across major highways (I-80, I-280, I-580, I-680, I-880, US-101, SR-237, SR-85), creating a high-dimensional multivariate structure. This spatial distribution enables analysis of spatial-temporal dependencies and network-wide correlation patterns revealing congestion propagation dynamics.

Key time series properties include autocorrelation reflecting traffic state persistence, heteroskedasticity with variance increase during congestion, and sudden regime changes between free-flow and congested states.

<br />

### Details

| Dataset info |                                                                                        |
|--------------|----------------------------------------------------------------------------------------|
| Codename     | traffic                                                                                |
| Source       | https://github.com/laiguokun/multivariate-time-series-data/tree/master                 | 
| Paper        | Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Arxiv'17)   |
| Frequency    | hourly                                                                                 |
| Size         | (17544, 20)                                                                            |


<br />

### Sample Plots


<p align="center">
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/traffic/imputegap_traffic_1_plot.jpg" width="45%" />
  <img src="https://github.com/eXascaleInfolab/ImputeGAP/raw/main/imputegap/datasets/docs/traffic/imputegap_traffic_plot.jpg" width="45%" />
</p>


<br /><br /><hr /><br /><br />





## References

<a name="ref1"></a>
[1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

[2] Ines Arous, Mourad Khayati, Philippe Cudré-Mauroux, Ying Zhang, Martin L. Kersten, Svetlin Stalinlov: RecovDB: Accurate and Efficient Missing Blocks Recovery for Large Time Series. ICDE 2019: 1976-1979

[3] Christopher Mutschler, Holger Ziekow, and Zbigniew Jerzak. 2013. The DEBS  2013 grand challenge. In debs, 2013. 289–294

[4] Mohammad Malekzadeh, Richard G. Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile Sensor Data Anonymization. In Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI ’19). ACM,  New York, NY, USA, 49–58. https://doi.org/10.1145/3302505.3310068
