# Crypto-Daily-Change-Prediction-and-Portfolio-Optimizer
In this project I created a script that analyzes data on various cryptocurrency pairs from Yahoo finance to predict daily price changes and optimize my Coinbase portfolio accordingly. 


First, the program downloads available data from Yahoo Finance on all of the crypto currency pairs that exchange to USD on Coinbase. Next, I created a class called "asset_window" that breaks all of the data down into set time frames for each asset. The window length can be adjusted to increase or decrease the initial number of features in the model. Additionally, the desired daily return, asset weight limit, lookback length, and rebalancing threshold can also be adjusted to alter the model's performance. 


From here, I separated these windows into an 85-15 train-test split to begin training the model on the historical data. I then set the log change in closing price on the last day of the window as the y-variable, and used the rest of the window prior to the last day as my X-variables. Next, I utilized an autoencoder for dimensionality reduction to avoid overfitting and allow the model to generalize better. I then used XGBoost's Extreme Gradient Boosted Trees Regressor to predict the change in closing price based on the latent space generated from the autoencoder. 


After this, I used cvxpy to optimize my portfolio based on various constraints. Because Coinbase currently does not allow shorting cryptocurrencies, I had to construct a buy-only portfolio, meaning no weight could be below 0. Additionally, I wanted to ensure that assets weren't being overweighted, so I included the aforementioned weight limit in my portfolio optimization. Beyond this, I set a risk allowance constraint to ideally increase the sharpe ratio of my portfolio. Finally, I added a constraint that the daily return must be higher than the desired daily return and set my objective function as maximizing the return of the portfolio. 


For the final section of this program, I created a dataframe to determine the portfolio's current weights, the desired weights based on the model, and the required trades needed to reconcile these differences. Finally, the program executes the trades using Coinbase's API.
