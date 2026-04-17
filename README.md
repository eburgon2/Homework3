# Homework3
This project aims to create a recurrent neural network model that can predict streamflow based
upon data from other sites. The model’s performance was analyzed for both training losses and
predictive accuracy. Overall, the model was determined to perform moderately well, but it would
be better to consider more sites or parameters for a clearer analysis.

The final Time Series generated for Weber River is shown below: 


<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/12fd2a17-f211-4739-b240-2b6a6987b61f" />


Overall, the LSTM model based upon 3 gauge sites performed moderately. It predicts well for
standard flow rates but will quickly fail with anomalistic data. Because it performed well at
lower flow rates, I do believe that the parameter selection was a good match for this model. So,
to address this issue, the model would need to be trained with a larger amount of gauge sites. I
would also recommend removing anomaly data at least at first, as including it right away can
lead to a mis-trained model. A final recommendation I have is to account for mass balance in
water catchments. There are clear physics laws that the model can use to train, and implementing
these can only increase accuracy. In summary, the LSTM for Weber River is a great example for
the faults of small and noisy LSTMs with good training settings. Good coding can help the
model train and reduce loss very well, but noisy or small data sets will lead to moderate precision
and poor predictions for extreme conditions.
