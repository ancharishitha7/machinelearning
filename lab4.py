import numpy as np

# Number of hidden states
N = int(input("Enter number of hidden states: "))

# Number of observation symbols
M = int(input("Enter number of observation symbols: "))

print("\nEnter Initial State Probabilities:")
pi = np.array(list(map(float, input().split())))

print("\nEnter Transition Matrix:")
A = []
for i in range(N):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

print("\nEnter Emission Matrix:")
B = []
for i in range(N):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    B.append(row)
B = np.array(B)

# Number of observation sequences
S = int(input("\nEnter number of observation sequences: "))

for s in range(S):

    print(f"\nObservation Sequence {s+1}")

    T = int(input("Enter length of sequence: "))

    print(f"Enter {T} observation indices (0 to {M-1}):")
    O = list(map(int, input().split()))

    alpha = np.zeros((T, N))

    # Initialization
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][O[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            total = 0
            for i in range(N):
                total += alpha[t-1][i] * A[i][j]
            alpha[t][j] = total * B[j][O[t]]

    # Termination
    probability = np.sum(alpha[T-1])

    print("\nForward Probability Table:")
    print(alpha)
    print("Likelihood =", probability)