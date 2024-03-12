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
## Adversial Policy
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/7fc76683-03eb-42d1-a648-19ad8711705d)

## Goal percentage of adversarial policy :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/cf577444-3984-4d04-b791-d9d8ad126bfc)
## Adversarial policy state-value function : 
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/eb496b0b-d88e-4a4b-a92e-35d2cbaf0b68)
## Policy after improvement :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/e848d032-d43d-4e1f-8718-509001a61800)
## Goal percentage of improved policy :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/bae8ef25-e7d9-49cd-a8ba-cd71e7621386)
## Improved policy state-value function :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/9673ae4b-3245-4ab3-83cf-49f1014a47de)
## Comparing the initial and the improved policy :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/af8e5d07-fab5-4784-b0b1-740ea656287c)
## Optimal policy (PI) :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/27654183-42a5-4149-972e-527d12a09dc5)
## Goal percentage of optimal policy :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/99833d4d-a064-497b-872e-02bd49028f2d)
## Optimal policy state-value function :
![image](https://github.com/Vivekreddy8360/policy-iteration-algorithm/assets/94525701/048b1f62-35b3-4eec-b2b0-bc7687b3333d)

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.
