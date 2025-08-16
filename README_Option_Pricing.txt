Option Pricing Using ResNet

Project Overview
This project leverages a ResNet(2,2) model to predict option prices. The model
supports both simulated data (Black-Scholes and Geometric Asian options) and real
market data to predict Option Prices in Bins.

Features
● Supports classification-based ResNet models.
● Allows users to choose between simulated and real market data.
● Outputs evaluation metrics: Error Metric, Inaccuracy Metric(rho)
● Bin width for classification is set to 0.1.

Prerequisites
● Python 3.x
● PyTorch
● NumPy
● Pandas
● Scikit-learn

Installation
pip install torch numpy pandas scikit-learn

Usage
1. Run the script:
   python resnet_option_pricing.py

2. Select data type:
   o 1: Real Market Data
   o 2: Simulated Data

3. If using real data: Make sure you have placed the real dataset CSV in the project
   folder (and updated the code to its filename if needed). The script by default expects
   realMarketData.csv

4. If using simulated data: After choosing 2 for simulated data, you will get a second
   prompt to select the option pricing model for simulation:

   -> Choose option type to simulate:
      1. Black-Scholes European Option
      2. Geometric Asian Option
      Enter 1 or 2:

   -> File named simulated_data_class.csv will be generated for your
      reference.

5. After training, the script will evaluate the model on the test set and produce outputs:
   It saves the model predictions and corresponding true values to a CSV file. By default,
   ● for real data, the file is named output_predictions.csv,
   ● for simulated data, the file is named predictions_output.csv (in the
     working directory).

6. The script prints out the evaluation metrics in the console in the following format:
   --- Evaluation Metrics ---
   Error Metric (EM): X.XXXXXX
   Inaccuracy Metric (RHO): Y.YYYYYY

7. A line plot window will be plotted showing the ‘True vs Predicted Bins’ for the first 100
   test samples.

8. Interpreting Results: After running, you can inspect:
   ● The CSV output file to see individual predictions. A small difference between
     True_Bin and Predicted_Bin (e.g., 0 vs 1) means the model was close,
     whereas large differences indicate bigger errors.
   ● The printed EM and RHO metrics: lower EM indicates more accurate overall
     pricing (on average), and lower RHO (closer to 0) means very few predictions are
     significantly wrong (off by more than 2 bins).
   ● The line plot to quickly gauge if the predicted trend aligns with the actual price
     trend for a subset of data.

Input Features:
For Real Market Data
● Underlying Assest Price(S)
● Strike Price(K)
● Time to Maturity(T)
● Implied Volatility
● Previous Settlement Price
● Change in Underlying asset Price(Delta_S)
● Turnover Rate of Underlying Asset price

For Simulated Data
● Underlying Asset Price
● Strike Price
● Time to Maturity
● Risk Free Interest Rate
● Volatility

Output Feature (Simulated & Real Market Data):
● Predicted bin of the option price

Output
● CSV file with true and predicted bins or continuous prices.

Contributors
● Vaibhav Ingle
