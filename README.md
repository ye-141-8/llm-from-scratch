# LLM from scratch

Learning how neural networks and LLMs work by building them step by step, from a single neuron to a tiny GPT.

## About this project

I'm a beginner learning Python and machine learning. I use Claude (an AI assistant) and other resources to help me learn and to explain concepts like neurons, weights and training. I change and test the code myself, and I work through it line by line until I understand it.

Found a mistake, or something that is hard to understand? Please open an issue. I'd like this to be easy for other beginners to learn from.

## Stages

- [x] **Stage 1** (`neuron.py`): a single neuron with one weight learns a rule like "output = 4 × input" from one example.
- [x] **Stage 2a** (`neuron2.py`): a neuron with two inputs, a bias and a sigmoid activation learns the OR rule. Switch the data line and it learns AND instead.
- [ ] **Stage 2b**: see why one neuron cannot learn XOR.
- [ ] **Stage 2c**: add a hidden layer so a small network can learn XOR.
- [ ] **Stage 3**: a tiny GPT in PyTorch.

## How to run

Stages 1 and 2 only need Python 3, with no extra libraries:

```
python neuron.py
python neuron2.py
```

## The idea in one line

The neuron guesses, we measure the error, we nudge the weights a little to make the error smaller, and we repeat this thousands of times.

## Words used in the code

| Word | Meaning |
|---|---|
| weight | a number saying how much one input matters |
| bias | a number that sets the neuron's starting lean toward yes or no |
| sigmoid | a function that squashes any number into the range 0 to 1 |
| error | how far the neuron's answer is from the correct answer |
| learning rate | how big each nudge is |
| epoch | one full pass through all the training examples |