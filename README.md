# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
## POLICY ITERATION ALGORITHM
The environment has 7 states: Two Terminal States: G: The goal state & H: A hole state. Five Transition states / Non-terminal States including S: The starting state.
## POLICY IMPROVEMENT FUNCTION
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s] [a]:
          Q[s] [a] += prob * (reward + gamma * V[next_state] * (not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi
```

## POLICY ITERATION FUNCTION
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    # Write your code here to implement the policy iteration algorithm
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
    while True:
      old_pi = {s:pi(s) for s in range (len (P))}
      V = policy_evaluation (pi, P, gamma, theta)
      pi = policy_improvement (V, P, gamma)
      if old_pi == {s:pi(s) for s in range (len(P)) }:
        break
    return V, pi
```

## OUTPUT:



## RESULT:

