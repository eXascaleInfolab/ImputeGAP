ImputeGAP Documentation
=======================

.. admonition:: 📢 News !

        - [11-12-2026] Version 1.1.3 is out! The new release includes the following major updates:
            - New LLMs algorithms
            - New parameter to select CPU or GPU execution for imputation algorithms.
        - [04-06-2025] ImputeGAP was be presented as a Tutorial at KDD 2025 in Toronto! `[Link] <https://imputegap-tutorials.github.io/KDD-2025/>`_


ImputeGAP is a comprehensive Python library for imputation of missing values in  time series data. It implements user-friendly APIs to easily visualize, analyze, and repair incomplete time series datasets. The library supports a diverse range of imputation algorithms and modular missing data simulation catering to datasets with varying characteristics. ImputeGAP includes extensive customization options, such as automated hyperparameter tuning, benchmarking, explainability, and downstream evaluation.


In detail, the library provides:
    - Over 40 state-of-the-art time series imputation algorithms from six different families
    - Several imputation univariate time series datasets and utilities to handle multivariate ones
    - Configurable contamination module that simulates real-world missingness patterns
    - AutoML techniques to parameterize the imputation algorithms
    - Unified benchmarking pipeline to evaluate the performance of imputation algorithms
    - Modular analysis tools to assess the impact of imputation on time series downstream tasks
    - Expainability module to understand the impact of time series features on the imputation results
    - Adjustable wrappers to integrate new algorithms in different languages


.. raw:: html

   <br>



.. _get_started:

Getting Started
---------------

.. raw:: html

   <script>
      function applyTheme(e)
      {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }

      const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
      applyTheme(darkQuery); // Apply on load
      darkQuery.addEventListener('change', applyTheme); // React to changes
   </script>


   <div class="index-navigation" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">

      <div class="card">
        <h3><a href="getting_started.html">🚀 Installation</a></h3>
        <p>Read the guide on how to install ImputeGAP on your system.</p>
      </div>

      <div class="card">
        <h3><a href="tutorials.html">📖 Full tutorials</a></h3>
        <p>Check the tutorials to learn how to use ImputeGAP.</p>
      </div>

      <div class="card">
        <h3><a href="imputegap.html">📦 API</a></h3>
        <p>Find the main API for each submodule in the index.</p>
      </div>

      <div class="card">
        <h3><a href="integration.html">🧩 Integration</a></h3>
        <p>Explore how contribute to the library.</p>
      </div>



.. raw:: html

   <br>



.. _algorithms:

Imputation Algorithms
---------------------

ImputeGAP integrates over 35 imputation algorithms, spanning the full spectrum of imputation method families:

🚧 = under integration


.. list-table::
   :header-rows: 1
   :widths: 36 30 34
   :class: algs-left

   * - **Algorithms**
     - **Families**
     - **Venues -- Years**
   * - MOMENT [37]_
     - LLMs
     - ICML -- 2024
   * - NuwaTS [33]_
     - LLMs
     - Arxiv -- 2024
   * - GPT4TS [34]_
     - LLMs
     - NeurIPS -- 2023
   * - LLM4HRS [44]_ 🚧
     - LLMs
     - IEEE Tr Geo RS -- 2025
   * - UniTS [42]_ 🚧
     - LLMs
     - NeurIPS -- 2024
   * - Timer [43]_ 🚧
     - LLMs
     - ICML -- 2024
   * - MissNet [27]_
     - Deep Learning
     - KDD -- 2024
   * - BitGraph [32]_
     - Deep Learning
     - ICLR -- 2024
   * - BayOTIDE [30]_
     - Deep Learning
     - ICML -- 2024
   * - MPIN [25]_
     - Deep Learning
     - PVLDB -- 2023
   * - TimesNet [36]_
     - Deep Learning
     - ICLR -- 2023
   * - SAITS [39]_
     - Deep Learning
     - ESWA -- 2023
   * - PRISTI [26]_
     - Deep Learning
     - ICDE -- 2023
   * - GRIN [29]_
     - Deep Learning
     - ICLR -- 2022
   * - CSDI [35]_
     - Deep Learning
     - NeurIPS -- 2021
   * - DeepMVI [24]_
     - Deep Learning
     - PVLDB -- 2021
   * - HKMFT [31]_
     - Deep Learning
     - TKDE -- 2021
   * - MRNN [22]_
     - Deep Learning
     - IEEE Trans on BE -- 2019
   * - BRITS [23]_
     - Deep Learning
     - NeurIPS -- 2018
   * - GAIN [28]_
     - Deep Learning
     - ICML -- 2018
   * - T1 [45]_ 🚧
     - Deep Learning
     - ICLR -- 2026
   * - FIM-ℓ [46]_ 🚧
     - Deep Learning
     - ICLR -- 2025
   * - SSD-TS [47]_ 🚧
     - Deep Learning
     - KDD -- 2025
   * - ImputeINR [48]_ 🚧
     - Deep Learning
     - IJCAI -- 2025
   * - CoFill [49]_ 🚧
     - Deep Learning
     - IJCAI -- 2025
   * - SADI [50]_ 🚧
     - Deep Learning
     - AAAI -- 2025
   * - LSCD [51]_ 🚧
     - Deep Learning
     - ICML -- 2025
   * - FGTI [52]_ 🚧
     - Deep Learning
     - NeurIPS -- 2024
   * - SSSD [53]_ 🚧
     - Deep Learning
     - TMLR -- 2023
   * - MTSIT [54]_ 🚧
     - Deep Learning
     - IEEE Signal Processing Letters -- 2022
   * - SSGAN [55]_ 🚧
     - Deep Learning
     - AAAI -- 2021
   * - PoGeVon [57]_ 🚧
     - Deep Learning
     - KDD -- 2023
   * - CDRec [1]_
     - Matrix Completion
     - KAIS -- 2020
   * - TRMF [8]_
     - Matrix Completion
     - NeurIPS -- 2016
   * - ROSL [4]_
     - Matrix Completion
     - CVPR -- 2014
   * - GROUSE [3]_
     - Matrix Completion
     - Arxiv -- 2010
   * - SoftImpute [6]_
     - Matrix Completion
     - JMLR -- 2010
   * - SVT [7]_
     - Matrix Completion
     - SIAM J. OPTIM -- 2010
   * - SPIRIT [5]_
     - Matrix Completion
     - VLDB -- 2005
   * - IterativeSVD [2]_
     - Matrix Completion
     - BIOINFORMATICS -- 2001
   * - TIDER [58]_ 🚧
     - Matrix Completion
     - ICLR -- 2023
   * - TKCM [11]_
     - Pattern Search
     - EDBT -- 2017
   * - STMVL [9]_
     - Pattern Search
     - IJCAI -- 2016
   * - DynaMMo [10]_
     - Pattern Search
     - KDD -- 2009
   * - PSW-I [59]_ 🚧
     - Pattern Search
     - ICLR -- 2025
   * - IIM [12]_
     - Machine Learning
     - ICDE -- 2019
   * - XGBOOST [13]_
     - Machine Learning
     - KDD -- 2016
   * - MICE [14]_
     - Machine Learning
     - Statistical Software -- 2011
   * - MissForest [15]_
     - Machine Learning
     - BioInformatics -- 2012
   * - KNNImpute
     - Statistics
     - _
   * - Interpolation
     - Statistics
     - _
   * - MinImpute
     - Statistics
     - _
   * - ZeroImpute
     - Statistics
     - _
   * - MeanImpute
     - Statistics
     - _
   * - MeanImputeBySeries
     - Statistics
     - _



.. raw:: html

   <br>



.. _cite:

Cite
----

If you like our library, please add a ⭐ in our `GitHub repository <https://github.com/eXascaleInfolab/ImputeGAP/>`_.

If you use ImputeGAP in your research, please cite the following papers:

.. code-block:: bash

    @article{nater2025imputegap,
      title = {ImputeGAP: A Comprehensive Library for Time Series Imputation},
      author = {Nater, Quentin and Khayati, Mourad and Pasquier, Jacques},
      year = {2025},
      eprint = {2503.15250},
      archiveprefix = {arXiv},
      primaryclass = {cs.LG},
      url = {https://arxiv.org/abs/2503.15250}


    @article{nater2025kdd,
      title = {A Hands-on Tutorial on Time Series Imputation with ImputeGAP},
      author = {Nater, Quentin and Khayati, Mourad and Cudré-Mauroux, Philippe},
      year = {2025},
      booktitle = {SIGKDD Conference on Knowledge Discovery and Data Mining (To Appear)},
      series = {KDD2025}
    }



.. raw:: html

   <br>

.. raw:: html

   <br><br>

.. _references:

References
----------

.. [1] Mourad Khayati, Philippe Cudré-Mauroux, Michael H. Böhlen: Scalable recovery of missing blocks in time series with high and low cross-correlations. Knowl. Inf. Syst. 62(6): 2257-2280 (2020)

.. [2] Olga G. Troyanskaya, Michael N. Cantor, Gavin Sherlock, Patrick O. Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman: Missing value estimation methods for DNA microarrays. Bioinform. 17(6): 520-525 (2001)

.. [3] Laura Balzano, Robert D. Nowak, and Benjamin Recht. "Online Identification and Tracking of Subspaces from Highly Incomplete Information." CoRR abs/1006.4046, 2010.

.. [4] Xianbiao Shu, Fatih Porikli, Narendra Ahuja: Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-Rank Matrices. CVPR 2014: 3874-3881

.. [5] Spiros Papadimitriou, Jimeng Sun, Christos Faloutsos: Streaming Pattern Discovery in Multiple Time-Series. VLDB 2005: 697-708

.. [6] Rahul Mazumder, Trevor Hastie, Robert Tibshirani: Spectral Regularization Algorithms for Learning Large Incomplete Matrices. J. Mach. Learn. Res. 11: 2287-2322 (2010)

.. [7] Jian-Feng Cai, Emmanuel J. Candès, Zuowei Shen: A Singular Value Thresholding Algorithm for Matrix Completion. SIAM J. Optim. 20(4): 1956-1982 (2010)

.. [8] Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon: Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction. NeurIPS 2016: 847-855

.. [9] Xiuwen Yi, Yu Zheng, Junbo Zhang, Tianrui Li: ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data. IJCAI 2016: 2704-2710

.. [10] Lei Li, James McCann, Nancy S. Pollard, Christos Faloutsos: DynaMMo: mining and summarization of coevolving sequences with missing values. 507-516

.. [11] Kevin Wellenzohn, Michael H. Böhlen, Anton Dignös, Johann Gamper, Hannes Mitterer: Continuous Imputation of Missing Values in Streams of Pattern-Determining Time Series. EDBT 2017: 330-341

.. [12] Aoqian Zhang, Shaoxu Song, Yu Sun, Jianmin Wang: Learning Individual Models for Imputation. ICDE (2019)

.. [13] Tianqi Chen, Carlos Guestrin: XGBoost: A Scalable Tree Boosting System. KDD 2016: 785-794

.. [14] Royston Patrick , White Ian R.: Multiple Imputation by Chained Equations (MICE): Implementation in Stata. Journal of Statistical Software 2011: 45(4), 1–20.

.. [15] Daniel J. Stekhoven, Peter Bühlmann: MissForest - non-parametric missing value imputation for mixed-type data. Bioinform. 28(1): 112-118 (2012)

.. [22] Jinsung Yoon, William R. Zame, Mihaela van der Schaar: Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks. IEEE Trans. Biomed. Eng. 66(5): 1477-1490 (2019)

.. [23] Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, Yitan Li: BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS 2018: 6776-6786

.. [24] Parikshit Bansal, Prathamesh Deshpande, Sunita Sarawagi: Missing Value Imputation on Multidimensional Time Series. Proc. VLDB Endow. 14(11): 2533-2545 (2021)

.. [25] Xiao Li, Huan Li, Hua Lu, Christian S. Jensen, Varun Pandey, Volker Markl: Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (Extended Version). Proceedings of the VLDB Endowment, 17(3):345–358, 2023.

.. [26] Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu: PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation. ICDE 2023: 1927-1939

.. [27] Kohei Obata, Koki Kawabata, Yasuko Matsubara, Yasushi Sakurai: Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series. KDD 2024: 2296-2306

.. [28] Jinsung Yoon, James Jordon, Mihaela van der Schaar: GAIN: Missing Data Imputation using Generative Adversarial Nets. ICML 2018: 5675-5684

.. [29] Andrea Cini, Ivan Marisca, and Cesare Alippi. "Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks." The Tenth International Conference on Learning Representations (ICLR), 2022.

.. [30] Shikai Fang, Qingsong Wen, Yingtao Luo, Shandian Zhe, Liang Sun: BayOTIDE: Bayesian Online Multivariate Time Series Imputation with Functional Decomposition. ICML 2024 (PMLR 235): 12993-13009.

.. [31] Liang Wang, Simeng Wu, Tianheng Wu, Xianping Tao, Jian Lu: HKMF-T: Recover From Blackouts in Tagged Time Series With Hankel Matrix Factorization. IEEE Trans. Knowl. Data Eng. 33(11): 3582-3593 (2021)

.. [32] Xiaodan Chen, Xiucheng Li, Bo Liu, Zhijun Li: Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values. ICLR 2024

.. [33] Jinguo Cheng, Chunwei Yang, Wanlin Cai, Yuxuan Liang, Qingsong Wen, Yuankai Wu: NuwaTS: a Foundation Model Mending Every Incomplete Time Series. Arxiv 2024

.. [34] Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin: One fits all: power general time series analysis by pretrained LM. NeurIPS 2023

.. [35] Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon: CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS 2021

.. [36] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023

.. [37] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski: MOMENT: A Family of Open Time-series Foundation Models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 (Proceedings of Machine Learning Research). PMLR OpenReview.net, 16115–16152.

.. [39] Wenjie Du, David Côté, Yan Liu : SAITS: : Self-attention-based imputation for time series . ESWA 2023

.. [42] Shanghua Gao, Teddy Koker, Owen Queen, Thomas Hartvigsen, Theodoros Tsiligkaridis, Marinka Zitnik : UniTS: A Unified Multi-Task Time Series Model. NeurIPS 2024

.. [43] Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, Mingsheng Long : Timer: Generative Pre-trained Transformers Are Large Time Series Models. ICML 2024

.. [44] Wang Shuyu, Li Wengen, Yang Hanchen, Guan Jihong, Liu Xiwei, Zhang Yichao, Qin Rufu, Zhou Shuigeng: LLM4HRS: LLM-based spatio-temporal imputation model for highly-sparse remote sensing data. IEEE Transactions on Geoscience and Remote Sensing. IEEE 2025

.. [45] Dongik Park, Hyunwoo Ryu, Suahn Bae, Keondo Park, Hyung-Sin Kim: T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation. ICLR (2026)

.. [46] Patrick Seifner, Kostadin Cvejoski, Antonia Körner, Ramsés J. Sánchez: Zero-shot Imputation with Foundation Inference Models for Dynamical Systems. ICLR (2025)

.. [47] Hongfan Gao, Wangmeng Shen, Xiangfei Qiu, Ronghui Xu, Bin Yang, Jilin Hu: SSD-TS: Exploring the Potential of Linear State Space Models for Diffusion Models in Time Series Imputation. KDD (2025)

.. [48] Mengxuan Li, Ke Liu, Jialong Guo, Jiajun Bu, Hongwei Wang, Haishuai Wang: ImputeINR: Time Series Imputation via Implicit Neural Representations for Disease Diagnosis with Missing Data. IJCAI (2025)

.. [49] Wenying He, Jieling Huang, Junhua Gu, Ji Zhang, Yude Bai: Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion. IJCAI (2025).

.. [50] Mohammad Rafid Ul Islam, Prasad Tadepalli, Alan Fern: Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios. AAAI (2025)

.. [51] Elizabeth Fons, Alejandro Sztrajman, Yousef El-Laham, Luciana Ferrer, Svitlana Vyetrenko, Manuela Veloso: LSCD: Lomb–Scargle Conditioned Diffusion for Time series Imputation. ICML (2025)

.. [52] Xinyu Yang, Yu Sun, Xiaojie Yuan, Xinyang Chen: Frequency-aware Generative Models for Multivariate Time Series Imputation. NeurIPS (2024)

.. [53] Juan Lopez Alcaraz, Nils Strodthoff: Diffusion-Based Time Series Imputation and Forecasting with Structured State Space Models. TMLR (2023)

.. [54] A. Yarkin Yıldız, Emirhan Koç, Aykut Koç: Multivariate Time Series Imputation With Transformers. IEEE Signal Processing Letters 29 (2022)

.. [55] Xiaoye Miao, Yangyang Wu, Jun Wang, Yunjun Gao, Xudong Mao, Jianwei Yin: Generative Semi-supervised Learning for Multivariate Time Series Imputation. AAAI (2021)

.. [57] Dingsu Wang, Yuchen Yan, Ruizhong Qiu, Yada Zhu, Kaiyu Guan, Andrew Margenot, Hanghang Tong: Networked Time Series Imputation via Position-aware Graph Enhanced Variational Autoencoders. KDD (2023)

.. [58] Shuai Liu, Xiucheng Li, Gao Cong, Yile Chen, Yue Jiang: Multivariate Time-Series Imputation with Disentangled Temporal Representations. ICLR (2023)

.. [59] Hao Wang, Zhengnan Li, Haoxuan Li, Xu Chen, Mingming Gong, Bin Chen, Zhichao Chen: Optimal Transport for Time Series Imputation. ICLR (2025)





.. raw:: html

   <br><br>



.. toctree::
   :maxdepth: 0
   :caption: Contents:
   :hidden:

   installation
   data_loading
   gapgenerator
   imputation
   benchmarking
   downstreamimpact
   explainability
   integration
   about_us
   imputegap


