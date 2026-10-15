# teach a neuron the rule that turns one input into one output, without telling it the rule
in_data = float(
    input("What is the input: ")
)  # input() gives text, float() makes it a number
target_data = float(
    input("What is the correct output: ")
)  # the answer we want it to learn

# an input of 0 would divide by zero below, so stop early with a message
if in_data == 0:
    raise SystemExit("The input can't be 0, please run again with another number.")

data = [
    (in_data, target_data)
]  # a list of (input, correct answer) pairs; here just one pair

w = 0.5  # the weight, starting as a bad guess

# learning rate: how big each nudge is
# it must stay below 2 / x^2 or the weight overshoots and blows up
# 0.5 / x^2 is safely inside that limit and makes the error halve each pass
lr = 0.5 / pow(in_data, 2)

# epoch = one full pass through all the data; it counts 0, 1, 2, ... 39
for epoch in range(25):
    # x gets the input and target gets the correct answer from each pair (x is not a counter)
    for x, target in data:
        guess = w * x               # the neuron's answer with its current weight
        error = target - guess      # how far off the answer is
        w += lr * error * x         # nudge the weight to shrink the error
    print(epoch, round(w, 4))       # show the weight after this pass
