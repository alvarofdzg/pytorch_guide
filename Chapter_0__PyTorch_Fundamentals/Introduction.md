### What is deep learning?
It's a subset of Machine learning.

### What is machine learning?
It's turning things (data) into numbers and finding patterns in those numbers.
It finds this patterns using code & math.

### How does a machine learning algorithm work?
In a machine learning algorithm you start with the inputs and the outputs, and the algorithm has to figure out the rules to get those outputs based on the inputs.

### Why should we use machine learning for a problem?
The reason is that for a complex problem can be very difficult to figure out the rules on your own. That's why we let that part to the algorithm.

You can use ML for literally anything as long as you can convert it into numbers and program it to find patterns. Literally it could be anything any input or output from the universe.

However, if you can build a simple rule-based system that doesn't require machine learning, do that (even if it's maybe not very simple).

### What deep learning is good for?
- Problems with long lists of rules - when the traditional approach fails, machine learning/deep learning may help.
- Continually changing environments - deep learning can adapt ('learn') to new scenarios.
- Discovering insights within large collections of data - can you imagine trying to hand-craft rules for what 101 different kinds of food look like?

### What deep learning is typically not good for?
- When you need explainability - the patterns learned by a deep learning model are typically uninterpretable by a human.
- When the traditional approach is a better option - if you can accomplish what you need with a simple rule-based system.
- When errors are unacceptable - since the outputs of deep learning model aren't always predictable.
- When you don't have much data - deep learning models usually require a fairly large amount of data to produce great results.

### Machine learning vs deep learning
Traditionally, you want to use machine learning algorithms on structured data: table of numbers.
One of the best algorithms for this type of data is "gradient boosted machine", such as XGBoost.
Common algorithms:
- Random forest.
- Gradient boosted models.
- Naive Bayes.
- Nearest neighbour.
- Support vector machine.
- ... many more.

Typically, for deep learning is better unstructured data: images, text, speech, etc.
One of the best algorithms for this type of data is "neural networks".
Common algorithms:
- Neural networks.
- Fully connected neural network.
- Convolutional neural network.
- Recurrent neural network.
- Transformer.
- ... many more.

### How does a neural network work?
Inputs -> Numerical encoding -> Learns representation (patterns/features/weights) -> representation outputs -> outputs

Each layer is usually combination of linear (straight line) and/or non-linear (not-straight line) functions.

### Types of Learning
- Supervised Learning: you have data & labels.
- Unsupervised & Self-supervised Learning: you have data but not labels.
- Transfer Learning: takes the patterns that one model has learned of a dataset and transferring it to another model.
- Reinforcement Learning: you have an environment and an agent that does actions in that environment, and you give rewards and observation based on the actions made by th agent.

### What is/why use Pytorch?
Pytorch is:
- Most popular research deep learning framework
- Write fast deep learning code in Python (able to run on a GPU/many GPUs).
- Able to access many pre-build deep learning models(Torch Hub/torchvision.models).
- Whole stack: preprocess data, model data, deploy model in your application/cloud.

Pytorch is used verywhere.

### What is a tensor?
A tensor is the numerical ecoding and the representation output of a model.

## What are we going to cover in this first module?
-Pytorch basics & fundamentals (dealing with tensors nad tensor operations).

## What are we going to cover later?
- Preprocessing data (getting it into tensors).
- Building and using pretrained deep learning models.
- Fitting a model to the data (learning patterns).
- Making predictions with a model (using patterns).
- Evaluating model predictions.
- Saving and loading models.
- Using a trained model to make predictions on a custom data.
