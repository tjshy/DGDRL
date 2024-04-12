To Review (3yoE):

Thank you for your inquiries regarding the details in our paper. Your questions have greatly assisted in refining our manuscript, and we apologize for any inconvenience caused.	

	Q1：Problem formulation
A1: Considering a portfolio with N assets over T time slots, the portfolio management task aims to maximize profit and minimize risk. The formal equtation is 


$$ 
\begin{aligned} w^{*}(t) & =\arg \max _{w(t)}(1-\tau) w^{\top}(t) y(t) \\ & =\arg \max _{w(t)} w^{\top}(t) y(t), \\ & \text { s.t. } \sum_{i=1}^{N} w_{i}(t)=1, w_{i}(t) \in[0,1], t=1, \cdots, T . \end{aligned}
 $$

w^{*}(t) represents the portfolio weight vector, w(t) represents the portfolio weight, y(t) represents the yield, and \tau represents the transaction cost.

We will add the above problem formulation in the ** revised version**. 

	Q2: The idea behind the design of DGDRL
A2:  Our motivation is to better capture **long- and short-term relations** to **directly optimize** the portfolio's performance. 

- **Existing graph-based relation models need a two-phase process**: 1) predict the stock's trend and then 2) optimize the weight by predicted result with algorithms like quadratic programming, which inevitably leads to suboptimal performance. In contrast, our model **directly learns the portfolio weights** by reward chasing in the DRL framework.
- **Existing DRL-based model faces computing efficiency challenges in directly learning the complex graph relations** in the reward learning procedure. Our work devise a multi-channel graph attention mechanism for both the policy head and critic head of the RL agent so that the model could be optimized efficiently and naturally by the DRL framework. 

We aim to obtain representations of temporal features, which is the idea behind utilizing LSTM-HA. We seek to elucidate the intrinsic long-term relations within enterprises and the short-term relations evolving with market changes. This justifies the adoption of both static and dynamic graph structures. The specific model design is inspired by state-of-the-art methods such as THGNN, AlphaStock and DeepTrader.

	Q3&Q4: The RL setting and reward setting

A3&A4: The state refers to the condition of the stock market, specifically the historical characteristics of stocks observable by investors. The r_t represents the stock returns on the day following the execution of an action, independent of evaluative metrics such as ARR, AVol, etc. We will augment our explanations regarding this aspect in the **revised version**.

	Q5&Q6: The order of stocks in Figure 5(a)
A5&A6: 1) The neighboring nodes refer to other stocks that have relations with the current stock. 2) The stocks are arranged in default order according to their names. For this part, the order is not significant. Figure 5 investigates whether there exists a consistency relationship between the relationships among stocks and the formulation of strategies, namely, whether the relationships among stocks can contribute to investment strategies.

	D5: The meaning of neighboring
A7:The threshold is determined artificially based on the degree of correlation among stocks. We artificially configured it. Its purpose is to avoid the formation of a fully connected graph where all stocks are correlated.	

	D6: The statistics of the datasets
A8: Our SSE50 dataset comprises 50 stocks, DOWS30 consists of 30 stocks, and NAS100 encompasses 96 stocks, as introduced in Section 4.1. We have a record for each stock for every trading day.

	D7: Not all baselines appear in Figure 3

A9: Thanks. In figure 3, we did not record the profit curves of some models with the lowest performance for the ease of viewing. We can add them in the next version of our paper. 