# Dynamic Graph-based Deep Reinforcement Learning with Long and Short-term Relation Modeling for Portfolio Optimization (DGDRL)


## Abstract
In this paper, we proposed a practical solution for portfolio optimization with long- and short-term relation augmentation.  We achieve this goal by **devising two mechanisms for naturally modeling the financial market**: 

- Firstly, we utilize the static and dynamic graphs to represent the long and short-term relations, which are then naturally represented by the proposed multi-channel graph attention neural network. 
- Secondly, compared with the traditional two-phase approach, forecasting equity’s trend and then weighting them by combinatorial optimization, we naturally optimize the portfolio decisions by devising a novel graph-based deep reinforcement learning framework, which could directly guide the model to converge to optimal rewards.

**Different from existing graph-based relation models are inevitably suboptimal as they require a two-phase process** (predict the trend and then optimize the weight) and  **existing DRL-based models face computing efficiency challenges in directly learning the complex graph relations** in the reward learning procedure, our work **contributes to a practical graph-based DRL solution for deploying data science technologies in the real world**. In addition, we have demonstrated that our work has been evaluated by industry-level environments. 

Our motivation is to better capture **long- and short-term relations** to **directly optimize** the portfolio's performance. 

- **Existing graph-based relation models need a two-phase process**: 1) predict the stock's trend and then 2) optimize the weight by predicted result with algorithms like quadratic programming, which inevitably leads to suboptimal performance. In contrast, our model **directly learns the portfolio weights** by reward chasing in the DRL framework.
- **Existing DRL-based model faces computing efficiency challenges in directly learning the complex graph relations** in the reward learning procedure. Our work devise a multi-channel graph attention mechanism for both the policy head and critic head of the RL agent so that the model could be optimized efficiently and naturally by the DRL framework. 

## Requirements
- PyTorch >= 1.7.0+cu111
- Python 3.8.16
- Numpy 1.23.5
- Pandas 1.5.3
- tqdm 4.66.1

Our code is run in GPU by default (CUDA 11.1+ here) , you can change the device into CPU if only CPU is available.

## Repository Structre

The 'code' directory contains the source code of our methodology, executable via the "main.sh". 
The 'data' directory contains a subset of the testing dataset utilized for code execution verification. Among these, the csv files contain historical feature data, 'd_relation.pkl' stores the graphs for short-term relations, and the pickle files store the graph for long-term relations. It is important to note that training is conducted on a daily basis, with the division between training and testing set on a daily basis as well.
The "result" directory contains profit curves resulting from model testing.

## Experiment Details
Our SSE50, DOW30, and NDX100 datasets are sourced from [yfinance](https://github.com/ranaroussi/yfinance). We utilize the PyTorch library to implement the model and used the Adam optimizer for model training. The training process was run on a server with 32G memory and single NVIDIA Tesla V100 GPU. For the full training set, entire training process needs about 12 hours. In terms of parameters, our specific settings are placed in [config.py](#code/config.py)
During the graph construction process, the look-back window length $l_{w}$ is set as 20, the $d_{\mathcal{N}}$ and $t_{\mathcal{N}}$ of neighbor threshold function $\mathcal{N}_g(\cdot)$ are both set as 1 and the threshold screening is set as 0.2. 

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

2) AVol is used to measure the annualized average risk of a strategy. The computational formula is AVol = $\mathop{std}({R}_T) \times \sqrt{{N}_y}$, where ${R}_T = \{{r}_T^1, {r}_T^2, \ldots,{r}_T^t \}$. ${r}_T^t$ represents the rate of return at $t$.

3) MDD is used to describe the worst possible scenario after buying a product. The formula is MDD = $-max \_{\tau \in[1, T]}\left(max \_{t \in[1, \tau]}\left(\frac{r_{t}-r_{\tau}}{r_{t}}\right)\right)$, where $\tau$ and $t$ represent two moments within period $T$, and ${r}\_{t}$ and ${r}\_{\tau}$ are the cumulative returns at these two moments, respectively. 

4) ASR is based on volatility to describe the extra return for taking risk. The computational formula is ASR = ARR $/$ AVol.

5) CR describes the extra return on risk based on maximum drawdown. The computational formula is CR = ARR $/$ abs(MDD).

6) IR measure the excess return of an investment compared to a benchmark. The computational formula is IR = ${R}_T \times \sqrt{{N}_y} / \mathop{std}({R}_T) $.

ARR, Avol, and MDD serve as the three foundational metrics, with ARR being the most crucial as achieving high returns is the ultimate investment objective for investors. ASR, CR, and IR, on the other hand, represent the three advanced metrics, which integrate both the model's investment returns and the associated risks. ARR, ASR, CR, and IR exhibit superior performance as their values increase, signifying better outcomes. Conversely, for AVol and abs(MDD), lower values reflect improved performance.


## Supplementary experiments
### 1. Model Cumulative Return
<br/>
<div align="center">
	<img src="/fig/dji_ret.png" width="49%">
	<img src="/fig/nas_ret.png" width="49%">
</div>

Figures above respectively depict the return curves during the testing period on the DOW30 and NDX100 datasets. It can be observed that, except for a small initial period, DGDRL also maintains a leading position in returns throughout the entire period. In a market with an upward trend and positive index returns, all methods are able to achieve good returns, but DGDRL can achieve higher returns compared to other models. This indicates that the high returns of DGDRL are not solely dependent on market conditions but are rather obtained through efficient information extraction and decision-making capabilities.

### 2. Portfolio Consistency 
<br/>
<div align="center">
	<img src="/fig/consistency.png" width="60%">

</div>

we examined the consistency of the portfolio with stock relationships (including both long- and short-term relations) , as illustrated in figures above, to investigate the contribution of relationships between stocks to portfolio decision-making. Figure (a) represents the quantity of pairs of stocks that have a relationship on investment days and both appear in the portfolio; both axes represent stocks. It can be observed that the majority of stocks in the portfolio have associated stocks, indicating that stock relations are important factors in the model's portfolio decision-making. However, this selection is not arbitrary. The model needs to choose a small subset of associated stocks with investment potential from the vast and complex network of stock relationships each day. Figure (b) provides statistics on the cumulative quantity of consistent and inconsistent portfolio-stock relations over the test period of one year. The orange curve represents the quantity of stocks with relations in the portfolio, while the blue curve represents the quantity of stocks without relations in the portfolio. From the quantity perspective, it is evident that the stocks selected in the model's investment decisions are mostly those with relations. Both long-term and short-term relations between stocks contribute positively to the model's effectiveness, guiding the model's portfolio decision-making.

### 3. Parameter Sensitivity
<br/>
<div align="center">
	<img src="/fig/sensitivity.png" width="60%">

</div>

we analyse the experimental results of parameter sensitivity on the portfolio management task on SSE50 dataset with various parameters in Figures above.
According to Figure (a), it is shown that the performance of our model progresses with the increase of length of the look-back window. However, after peaking at 20, the model starts to fluctuate and degrade, since too long window leads to redundant useless information and too short window leads to inadequate useful information. Moreover, the length of the window directly determines the training cost, which is also a significant consideration factor. According to Figure (b), the performance of our model steadily improves with the increase of the number of attention heads, with the peak of 8. Because an appropriate number of attention heads can facilitate the extraction of sequential information for the model. 
According to Figure (c), we can observe that the performance of our model increases as the the dimension of hidden embedding increases, with the best dimension being 128. Continuing to increase the dimension won't bring more benefit to the model, since keep increasing the dimension will lead to high risks of overfitting, causing sub-optimal performance.
According to Figure (d), the most suitable number of Agent head layers is 8, and increasing or decreasing the number will both cause the model deterioration. This is because Agent-head needs a suitable number of layers to balance the information gain and noise generation.