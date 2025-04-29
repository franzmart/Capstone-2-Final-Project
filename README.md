## Using Machine learning models to forecast electricity prices in Houston, Texas. 

**Franz Martinez**
### Executive summary

**Project overview and goals:** The goal of this project is to develop a model to forecast the day ahead electricity price in Houston, Texas. The electricity prices change every hour and their day-ahead forecast is a key input for electricity traders before buying and selling contracts. We will obtain the model data from Electric Reliability Council of Texas (ERCOT), which is a USA government agency accountable to conciliate the power demand and supply in Texas. Additionally, we will train and test three different machine learning models to forecast the day ahead electricity prices, and evaluate the models based on Root-mean-square deviation (RMSE). Lastly, we will select the machine learning model with the lowest RMSE to forecast electricity prices.


**Findings:** Based on the ERCOT data for electricity prices for 2022, 2023, and 2024, the best autoregressive model for forecasting electricity prices for Houston, Texas is the Light Gradient Boosting Machine (LGBM) model, with an RMSE error of 0.06273. The nature of electricity prices is changing and its characterization by an autoregressive model is challenging. The electricity prices have value changes within years, months, weeks, and even days because of the electricity consumer behavior. Therefore, all those variables were accounted in the LGBM autoregressive model to increase the accuracy of the price forecasts. 

As for the other models, Extreme Gradient Boosting (XGBoost) model had the second greatest performance with an RMSE error of 0.06397, and the Random Forest model with an RMSE error of 0.07067. XGBoost and Random Forest were optimized by testing different sets of parameters (GridSearchCV). However, the optimal set of parameters for XGBoost and RandomForest were not powerful enough to minimize the errors and surpass the precision of the LGBM model.


**Results and conclusion:** Our evaluation of the best model returned the LGBM model as the most appropriate for the electricity prices in Houston, Texas. This model was the one with the lowest RMSE (0.06356) in the validation electricity dataset. We did a simple test and estimated the day-ahead price for 04/18/2025 at noon by using the LGMBM model. The forecast value was 28.35, and the actual price was 30.29. The precision is acceptable, less than US$ 2.


**Future research and development:** The autoregressive models are susceptible to structural changes in the behavior of economic actors. In the case of electricity prices, prices are determined by the supply and demand of power by Houston city. The next step will be to explore structural changes in the supply or demand of electricity to confirm that the machine learning models are processing historical data with no structural changes or economic distortions. For instance, in the recent years, the energy supply blend for Houston city had an structural change as fossil fuels are not the main source of power anymore, but wind and solar energy. We already included wind and solar power in our model.

**Next steps and recommendations:** The LGBM model offers a fair forecast of the electricity prices in Houston. However, severe weather can create a temporal distortion in electricity prices. For instance, tornadoes and hurricanes are common in Houston and they can damage the electricity distribution grids and consequently boost the electricity prices by limiting the power supply. The weather forecast should be reviewed before running the model.

Similarly, the model relies on the assumption that all the physical infrastructure is available to supply energy in Houston. However, this is not always the case, as the grid may be shut down temporarily due to grid maintenance. The grid availability should be reviewed before running the model.

### Rationale

The problem this project tries to solve is to forecast the day-ahead electricity price in Houston. This is the main input for traders before starting the trading journey in the morning. Depending on the forecast price values, traders can adjust their positions to maximize profits when they are buying and selling energy contracts. 

Accurate price forecasts are key to deploy energy trading strategies successfully, avoid unnecessary risks, and maximize profits. 

### Research Question

The question this project aims to answer is what are the best forecast electricity prices in Houston, Texas.

### Data Sources

**Dataset:**The datasets used in this project is sourced from the ERCOT website and can be accessed at 

Electricity Prices in Houston:
https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD

Houston Electricity Demand:
https://data.ercot.com/data-product-archive/NP6-346-CD

Wind Electricity Supply:
https://www.ercot.com/mp/data-products/data-product-details?id=NP4-732-CD

Solar Electricity Supply:
https://www.ercot.com/mp/data-products/data-product-details?id=np4-745-cd

The data are historical hourly information of prices, energy demand, wind supply, and solar supply from 1 January 2022 to 31 December 2024. All the four datasets were consolidated in one unique dataset with 26,302 rows.


19,724 random samples from the data are used for model training to facilitate computation, and the rest of samples (6,574) were used to test the model.

**Exploratory data analysis: ** There are no null values in this data, and we observed several numeric fields registered as string data type fields. We converted the price and hour columns to float values, and the date field to a date data type.


**Cleaning and preparation:** The unique ID column was dropped because it adds no meaningful information to the analysis. The "class" column was renamed "suicide", and its values numerically represented by designating the "suicide" class as 1 and the "non-suicide" class 0. 

The data is randomly split into train and test sets to facilitate holdout cross validation, with a test size of 0.25. 

**Preprocessing:** Before starting the outlier cleansing, we decided to analyze the Houston electricity price per year. For such purposes, we created four new date fields: Day, Month, Year, and DayandMonth. Once the information was ready, we created a line plot for day ahead electricity prices per month for 2022, 2023, and 2024 years. From the plot, price data showed peaks demonstrating a clear seasonality change for each year. These seasonal changes were due to weather factors and the heat waves leading to increased electricity demand and as a result prices. We need more granular analysis to define the right features for this model.

As part of our analysis of price behavior, we created a line plot for day ahead electricity prices per hour for four months (March, April, June, and July) of the training data for 2022, 2023, and 2024. From the plot, we observed that the price was doubled peaked in the evenings at 4:00 p.m. and 7:00 p.m. The forecast model is required to capture these effects. The price is clearly hour dependent. We therefore must use the built-in Hour feature. 

We can further refine the above analysis by drilling down to days of the week. We prepared a plot to display the average hourly price for weekday and weekends, computed for 4-month training data. From the diagram, the pattern of prices is almost the same in terms of peak hours for weekdays and weekends. However, the weekdays have different levels of price data in the peak area.

These observations lead to the following feature engineering configuration of our Machine Learning Price Forecast model:

1. Weekday and Week Number as built-in features or predictors to the machine learning model to indicate the days of the week.
2. Hour and Is Working Day built-in to reflect other patterns within a day.
3. Year as a built-in feature or predictor to the machine learning model to indicate the year.

**Final Dataset:** The final dataset consists of the following variables:

Lagged Variables - Previous Price, which is the previous hour electricity price.
Built-in Predictors - Hour, Weekday (with dummies), Week Number, Month, Is Working Day.
External Time Series Predictors - Demand, Wind supply, and Solar supply 

As for removing outliers, we prepared an electricity price histogram and observed prices beyond 100 dollars. Those prices were uncommon, and therefore, we removed them as outliers. This filter was applied for the Electricity Price and Previous Price fields. Additionally, we applied the log function to the Electricity Price and Previous Price fields to smooth them.

Similarly, we prepared an electricity demand histogram and observed quantities beyond 20,000 and below 7500. Those demand values were removed as outliers

Additionally, we prepared a wind and solar supply histograms and observed quantities beyond 20,000 and 25,000 respectively. Those supply values were removed as outliers from each supply field.

Lastly, we set our target variable as Electricity Price and started our correlation analysis and t test for each predictor variable. We observed the following variables as significant predictors.

1. Hour
2. Previous_Price
3. Houston demand
4. Wind supply
5. Solar supply
6. Week_number  
7. Year
8. Month
9. Is_Working_day
10. Monday (Dummy)
11. Friday (Dummy)
12. Saturday (Dummy)
13. Sunday (Dummy)

### Methodology

We prepared three models: Random Forest, XGBoost, and LGBM. Each of the models were optimized and then compared by the RMSE indicator. The model with the lower RMSE will be selected as the optimal forecast model. 

Models were trained on the training set and validated with the test set. Additionally, GridSearchCV was used to calibrate the models using the RMSE accuracy score.

Three models were trained, fine-tuned, and will be later compared to find the best model for this task.

**Random Forest Model:** A pipeline object is created to standardize the data by using StandardScaler and instantiate a Random Forest Regressor model. GridSearchCV is used to find (1) the optimal tree depth, with the options being [None, 5, 10], (2) the optimal minimum number of samples for a split, with the options being [2, 5, 10], (3) the optimal min number of samples for a leaf, with the options being [1, 2, 4]. The best model has no max tree, the minimum number of samples of a split is 2, and the minimum number of samples for a leaf is 2.


**Light Gradient Boosting Machine Model:** A pipeline object is created to standardize the data by using StandardScaler and instantiate a LGBM model. GridSearchCV is used to find (1) the optimal number of trees to train the model, with the options being [100, 200, 300, 400, 500], (2) the optimal learning rate, with the options being [0.01, 0.05, 0.1], (3) the optimal maximum tree depth, with the options being [3, 5, 7, 9, 11]. The best model has a tree number of 500, a learning rate of 0.1, and a maximum tree depth of 9.


**Extreme Gradient Boosting Model:** A pipeline object is created to standardize the data by using StandardScaler and instantiate a XGBoost model. GridSearchCV is used to find (1) the optimal number of trees to train the model, with the options being [100, 200, 500], (2) the optimal learning rate, with the options being [0.01, 0.1, 0.3, 0.5], (3) the optimal maximum tree depth, with the options being [3, 5, 7]. The best model has a tree number of 500, a learning rate of 0.1, and a maximum tree depth of 5.


### Model evaluation and results 

Model precision was evaluated by the RMSE error of each model. The model with the lowest RMSE error is LGBM with a RMSE value of 0.06273. The second model with lower RMSE error is XGBoost with a RMSE value of 0.06397. Lastly, the model with the highest RMSE error is Random Forest with a RMSE value of 0.07067. 

A detailed interpretation and evaluation of the best model can be found in the results and conclusion section of the executive summary above. 


### Jupiter File 

[Link] https://github.com/franzmart/Capstone-1---Initial-Report-and-EDA

### Contact and Further Information

Franz Martinez

Email: fmbowarte@gmail.com 

[LinkedIn](https://www.linkedin.com/in/franz-martinez-auditor/)
