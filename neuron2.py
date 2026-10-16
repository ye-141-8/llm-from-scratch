# a neuron with two inputs, a bias and an activation, learning the OR rule
# OR rule: the answer is 1 if at least one input is 1, otherwise 0

# "import" loads ready-made toolkits: math for the exponent, random for a random start
import math
import random

# training data: each item is ((input1, input2), correct answer)
# a neuron with two inputs, a bias and an activation, learning the OR rule
# (to learn the AND rule instead, switch which "data" line below is commented out)

# OR Rule/Concept:
data = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]

# AND Rule/Concept:
# data = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]

# "def" defines a reusable recipe; "return" hands the result back
# sigmoid squashes any number into the range 0..1 (the "how strongly it fires" part)
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# one weight per input, starting at a random number between -1 and 1
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = 0.0  # the bias, starts at 0

lr = 1.0  # learning rate: size of each nudge

# one epoch = one full pass over all 4 examples (so 4 nudges per epoch now)
for epoch in range(5000):
    total_error = 0  # adds up how wrong the neuron was during this pass

    # each item is ((x1, x2), target), so this line splits it into three variables at once
    for (x1, x2), target in data:
        # weighted sum of inputs plus bias, squashed by sigmoid
        out = sigmoid(w1 * x1 + w2 * x2 + b)
        error = target - out
        total_error += (
            error**2
        )  # ** means "to the power of"; squaring makes it positive

        # how much to nudge: the error, scaled by how steep the sigmoid is right now
        delta = error * out * (1 - out)

        # nudge each weight and the bias
        # (if an input is 0, its weight gets no nudge, because delta is multiplied by 0)
        w1 += lr * delta * x1
        w2 += lr * delta * x2
        b += lr * delta
        
        # add the line below to see what the inputs(x1,x2) and out(sigmoid function) 
        # and delta(nudge amount) is in each time the loops runs
        # also make sure to change the range in the first loop so its shorter data for test.
        # print(x1, x2, round(out, 3), round(delta, 4))

    # % gives the remainder; epoch % 1000 == 0 is true only every 1000th pass
    if epoch % 1000 == 0:
        # round(total_error, 4) means that round total_error and keep 4 decimals
        print(epoch, round(total_error, 4))         
    
# test the trained neuron on all four examples
print("weights:", round(w1, 3), round(w2, 3), "bias:", round(b, 3))
for (x1, x2), target in data:
    print(
        (x1, x2),
        "correct:",
        target,
        "neuron says:",
        round(sigmoid(w1 * x1 + w2 * x2 + b), 3),
    )
