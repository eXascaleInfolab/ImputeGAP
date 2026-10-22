# IMPUTATION DEFAULT VALUES ==========================================================================

OPTI_DICTIONARY = {
    "cdrec-rank": [2, 3, 4, 6, 8, 10, 20],
    "iterative_svd-rank": [2, 3, 4, 6, 8, 10, 20],
    "grouse-max_rank": [2, 3, 4, 6, 8, 10, 20],
    "rosl-rank": [2, 3, 4, 6, 8, 10, 20],
    "soft_impute-max_rank": [2, 3, 4, 6, 8, 10, 20],
    "spirit-k": [2, 5, 10],
    "svt-tau": [0.1, 0.2, 0.4, 0.6, 0.8],
    "trmf-K": [1, 2, 3, 4, 7],

    "stmvl-window_size":[2, 3, 4, 6, 7, 8, 10],
    "dynammo-h":[3, 4, 5, 6, 8, 10, 20],
    "tkcm-rank":[1, 5, 10],

    "iim-learning_neighbors": [2, 3, 4, 6, 8, 10],
    "mice-initial_strategy": ["mean", "median", "most_frequent", "constant"],
    "miss_forest-n_estimators": [10, 25, 50, 100],
    "xgboost-n_estimators": [2, 3, 4, 6, 8, 10],

    "mrnn-hidden_layers": [32, 64, 108, 256, 512],
    "brits-hidden_layers": [32, 64, 108, 256, 512],
    "deep_mvi-lr": [0.001, 0.01, 0.1],
    "mpin-window": [2, 4, 6, 8, 10],
    "miss_net-n_components": [10, 15, 20, 30],
    "gain-alpha": [10, 50, 100, 150],
    "grin-alpha": [2, 5, 10, 20, 50],
    "bay_otide-K_trend": [10, 25, 30, 50],
    "bit_graph-subgraph_size": [2, 5, 10],
    "hkmf_t-epochs": [2, 5, 10],
    "saits-n_head": [2,4,8],
    "timesnet-top_k": [2, 3, 5, 7],
    "csdi-beta": [[0.01, 0.1], [0.001, 0.1], [0.0001, 0.2], [0.0001, 0.5]],
    "pristi-beta": [[0.01, 0.1], [0.001, 0.1], [0.001, 0.2], [0.0001, 0.2]],
    "nuwats-gpt_layers": [3,6],
    "gpt4ts-gpt_layers": [2,3,6],
    "moment-model_size": ["small", "medium", "large"],
}