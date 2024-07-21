# Predicting Traffic Congestion

**Reed R. Hedderman**

## Abstract

Traffic lights create about 20% of the traffic issues we have today. What if we could create a model that could predict the traffic patterns for intersections to best optimize the traffic signal phases and timing? This project discusses a machine learning model that will predict the traffic patterns for different intersections split in different ways to analyze the best way to predict and optimize traffic. We will look at three different intersections: one will be split by week, the next split by day, and the third split by hour. This will give us a good representation of how the traffic patterns look over time. We will be using a GRU neural network model since we are dealing with a time series dataset.

## Index Terms

- GRU Neural Network
- Root Squared Mean Error
- Traffic Patterns
- Lachner’s Theory
- ITSCP

## Introduction

Inner city traffic has been an issue for as long as traffic lights and cars have been in cities. The problem of sitting through numerous traffic lights, wondering why this red light is taking so long, and knowing you’re going to be late is what my project intends to solve. Researchers are constantly working on this issue, but their focus is more on highway traffic and not as much on inner city traffic or traffic lights.

To solve this issue, I have created a machine learning model that will predict traffic patterns. I used a recurrent neural network as the model to predict the traffic on three different intersections in Austin, Texas. Since my data is a time series, or sequence of data organized by time, I used a recurrent neural network since they can exploit the sequential nature of the data and learn from the temporal dependencies.

## Methodologies

### Describe the methods/models used in your research

The main method/model utilized in this research is the Gated Recurrent Unit (GRU) neural network. The GRU is a type of recurrent neural network (RNN) that has shown promising results in sequence prediction tasks. It is particularly suitable for modeling sequential data with long-term dependencies. The GRU neural network architecture consists of recurrent units with gating mechanisms that help the network selectively update and reset the hidden state, allowing it to capture and retain relevant information over time.

### Mathematical model description

At each time step t, the GRU takes an input vector x(t) and the previous hidden state h(t-1). It applies linear transformations to the input and hidden state using trainable weights and biases. These transformations are represented by the following equations:

1. **Reset Gate (r(t))**:  
   `r(t) = sigmoid(Wr * x(t) + Urr * h(t-1) + br)`  
   The reset gate determines how much of the previous hidden state to forget and is computed by applying the sigmoid activation function to the linear combination of the input, previous hidden state, and corresponding biases.

2. **Update Gate (z(t))**:  
   `z(t) = sigmoid(Wz * x(t) + Uzr * h(t-1) + bz)`  
   The update gate controls how much of the new input should be incorporated into the current hidden state and is calculated using the sigmoid function applied to the linear combination of the input, previous hidden state, and biases.

3. **Candidate Activation (ĉ(t))**:  
   `ĉ(t) = tanh(Wc * x(t) + Ur * (r(t) * h(t-1)) + bc)`  
   The candidate activation represents the new information to be added to the hidden state. It is computed by applying the hyperbolic tangent (tanh) activation function to the linear combination of the input, the reset gate multiplied by the previous hidden state, and the corresponding biases.

4. **Hidden State (h(t))**:  
   `h(t) = (1 - z(t)) * h(t-1) + z(t) * ĉ(t)`  
   The hidden state is updated based on the values of the update gate and the candidate activation. It combines the previous hidden state with the new information, weighted by the update gate. This combination allows the GRU to selectively retain and update relevant information over time.

These equations are applied sequentially for each time step in the input sequence, allowing the GRU to capture the temporal dependencies and learn patterns in the data. The final hidden state at the last time step can be used for making predictions or further processing, depending on the specific task.

By training the GRU neural network using suitable optimization algorithms, such as stochastic gradient descent (SGD) with backpropagation, the model learns to adjust the weights and biases to minimize a chosen loss function (e.g., mean squared error) between the predicted outputs and the ground truth values. This training process enables the GRU to effectively learn and make accurate predictions for traffic forecasting.

### Model Schema

The model consists of multiple GRU layers (GRU_1, GRU_2, ..., GRU_5) stacked on top of each other. Each GRU layer takes the input from the previous layer and produces an output that is passed to the next layer. The final output is obtained from the last GRU layer and is connected to a dense layer (Dense_1), which produces the predicted values. This model schema allows the GRU network to learn and capture the temporal dependencies in the input sequence, enabling accurate traffic prediction.

## Datasets

The data set used was taken from the Official City of Austin open data portal titled “Camera Traffic Counts.” This data set showed the most current traffic data on intersections in Austin, Texas and contained all the information needed. The data includes the date and time each row was measured, the volume of cars in that intersection every fifteen minutes, the direction they were headed, and how long on average the vehicles were in a specific intersection zone.

## Experimental Results

In the initial analysis of the data, I looked at the current trends and traffic patterns. The data was divided into different categories: Year, Month, Week, Day, and Hour. The following images show the volume of traffic for three different intersections by Year, Month, and Date Number. The next images show the traffic trends by the Day of the week and by the hour.

To transform the data, it was normalized and then differenced. Intersection 0 was differenced weekly, intersection 2 was differenced daily (the days of the week), and intersection 3 was differenced by hour. The transformation made the data more linear and easier to gauge and predict.

After training the model, the accuracy of the predictions was assessed. The results showed that the shorter the time span, the more accurate the model was. From weekly to daily to hourly, the model accuracy improved.

## Discussions

The GRU model was efficient because it is designed to address the vanishing gradient problem. After normalizing the data and differencing it by week, day, and hour, an augmented Dickey-Fuller test ensured the series was stationary. The GRU model was also implemented with an early stopping function to prevent overfitting. The results showed that the less time between the difference in data, the more accurate the model is. The hourly predictions were the most accurate and had the lowest root mean squared error.

## Conclusions

Although this started out as a project to predict traffic light signals, the main objective was achieved. After finding an available dataset and cleaning it, a basic GRU neural network model was created to predict the traffic patterns. With a machine learning model that can predict traffic patterns at intersections by the week, day, or hour, city officials and traffic workers can optimize traffic light signal phases and timing.
