# Dynamic Graph-based Deep Reinforcement Learning with Long and Short-term Relation Modeling for Portfolio Optimization (DGDRL)


## Abstract
In this paper, we proposed a practical solution for portfolio optimization with long- and short-term relation augmentation.  We achieve this goal by **devising two mechanisms for naturally modeling the financial market**: 

- Firstly, we utilize the static and dynamic graphs to represent the long and short-term relations, which are then naturally represented by the proposed multi-channel graph attention neural network. 
- Secondly, compared with the traditional two-phase approach, forecasting equity’s trend and then weighting them by combinatorial optimization, we naturally optimize the portfolio decisions by devising a novel graph-based deep reinforcement learning framework, which could directly guide the model to converge to optimal rewards.

**Different from existing graph-based relation models are inevitably suboptimal as they require a two-phase process** (predict the trend and then optimize the weight) and  **existing DRL-based models face computing efficiency challenges in directly learning the complex graph relations** in the reward learning procedure, our work **contributes to a practical graph-based DRL solution for deploying data science technologies in the real world**. In addition, we have demonstrated that our work has been evaluated by industry-level environments. 

Our motivation is to better capture **long- and short-term relations** to **directly optimize** the portfolio's performance. 

- **Existing graph-based relation models need a two-phase process**: 1) predict the stock's trend and then 2) optimize the weight by predicted result with algorithms like quadratic programming, which inevitably leads to suboptimal performance. In contrast, our model **directly learns the portfolio weights** by reward chasing in the DRL framework.
- **Existing DRL-based model faces computing efficiency challenges in directly learning the complex graph relations** in the reward learning procedure. Our work devise a multi-channel graph attention mechanism for both the policy head and critic head of the RL agent so that the model could be optimized efficiently and naturally by the DRL framework. 

## Problem formulation

Considering a portfolio with N assets over T time slots, the portfolio management task aims to maximize profit and minimize risk. The formal equtation is 
The optimal weights $w^*(t)$ are defined as follows:

$$
w^{*}(t) = \arg \max _{w(t)}(1-\tau) w^{\top}(t) y(t)
$$

This can be simplified under the assumption that \( 1-\tau \) is a constant multiplier:

$$
w^{*}(t) = \arg \max _{w(t)} w^{\top}(t) y(t),
$$

subject to the constraints:

$$
\sum_{i=1}^{N} w_{i}(t)=1, \quad w_{i}(t) \in [0,1], \quad t=1, \cdots, T.
$$

w(t) represents the portfolio weight, y(t) represents the return of the stock at time t, and \tau represents the transaction cost.

## Requirements
- PyTorch >= 1.8.1+cu111
- Python 3.8.5
- Numpy 1.19.5
- nltk 3.6.1
- tqdm 4.60.0

Our code is run in GPU by default (CUDA 11.1+ here) , you can change the device into CPU if only CPU is available.

## Experiment Details
Our SSE50, DOW30, and NDX100 datasets are sourced from [yfinance](https://github.com/ranaroussi/yfinance). We utilize the PyTorch library to implement the model and used the Adam optimizer for model training. The training process was run on a server with 32G memory and single NVIDIA Tesla V100 GPU. For the full training set, entire training process needs about 12 hours. In terms of parameters, our specific settings are placed in [config.py](#config.py)
During the graph construction process, the look-back window length $l_{w}$ is set as 20, the $d_{\mathcal{N}}$ and $t_{\mathcal{N}}$ of neighbor threshold function $\mathcal{N}_g(\cdot)$ are both set as 1 and the threshold screening is set as 0.2. 

## Baselines and Metrics Description
The baseline methods we compared against are as follows:

- BLSW is a classical mean reversion strategy. It posits that stocks exhibiting a prolonged downtrend in the past are more investment-worthy.
- CSM is a classical momentum strategy. It posits that stocks showing a sustained upward trend in the past are more investment-worthy.
- Transformer is a widely used deep learning model based on self-attention mechanisms.
- TRA is a deep learning approach that classifies and predicts stocks based on the study of their  characteristics.
- CGM is a deep learning method that utilizes long-term and short-term  stock relationships for predicting stock trading volume.
- THGNN is a deep learning method that constructs short-term heterogeneous graphs for stock prediction.
- FactorVAE is a factor model based on prior-posterior learning approach for stock prediction.
- CTTS is a deep learning method based on CNN and Transformer for long-term modeling of time series features.
- PPO is a reinforcement learning method that uses constraints to ensure the stability of policy updates, thereby improving training efficiency. It is also one of the fundamental algorithms in Finrl, which is the first open-source DRL framework to demonstrate the great potential of financial reinforcement learning.
- Alphastock is a deep reinforcement learning method that constructs relationships between stocks based on attention mechanisms for portfolio management.
- DeepTrader is a deep reinforcement learning model that models market sentiment for controlling investment risk in portfolio management.
- DeepPocket is a deep reinforcement learning model that utilizes graph convolutional networks for portfolio optimization based on static relations of stocks.

We put baseline's parameter Settings in [baseline_setting.md](#baseline_setting.md)

The metrics used in our experiments are as follows:

1) ARR is an annualized average of return rate. The computational formula is ARR = ${r}_T \^{} \frac{{N}_y}{{N}_T}$, where  ${r}_T$ represents the cumulative returns within the period $T$, ${N}_y$ represents the number of trading days in a year, and ${N}_T$ represents the number of trading days in $T$.

2) AVol is used to measure the annualized average risk of a strategy. The computational formula is AVol = $std({R}_T) \times \sqrt{{N}_y}$, where ${R}_T = \{{r}_T^1, {r}_T^2, \ldots,{r}_T^t \}$. ${r}_T^t$ represents the rate of return at $t$.

3) MDD is used to describe the worst possible scenario after buying a product. The formula is
MDD = $ - \max \limits_{\tau \in [1,T]} $ ( $\max \limits_{t \in [1,\tau]} \ (\frac{{r}_t - {r}_\tau} {{r}_t})$ ), where $\tau$ and $t$ represent two moments within period $T$, and ${r}_t$ and ${r}_\tau$ are the cumulative returns at these two moments, respectively.

4) ASR is based on volatility to describe the extra return for taking risk. The computational formula is ASR = ARR $/$ AVol.

5) CR describes the extra return on risk based on maximum drawdown. The computational formula is CR = ARR $/$ abs (MDD).

6) IR measure the excess return of an investment compared to a benchmark. The computational formula is IR = ${R}_T \times \sqrt{{N}_y} / std({R}_T) $.

ARR, Avol, and MDD serve as the three foundational metrics, with ARR being the most crucial as achieving high returns is the ultimate investment objective for investors. ASR, CR, and IR, on the other hand, represent the three advanced metrics, which integrate both the model's investment returns and the associated risks. ARR, ASR, CR, and IR exhibit superior performance as their values increase, signifying better outcomes. Conversely, for AVol and abs(MDD), lower values reflect improved performance.

## Repository Structre

   The 'data' directory contains the data for our three real-world datasets. The csv files contain historical feature data, 'relation.pkl' stores the raw graphs for short-term relations, and 'relation2.pkl' holds the final graphs for short-term relations after threshold filtering. The npy files store the graph for long-term relations. It is important to note that training is conducted on a daily basis, with the division between training and testing set on a daily basis as well.





