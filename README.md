# Reinforcement Learning for Text-Based Home World

A reinforcement learning project that explores and compares **Tabular Q-Learning**, **Linear Function Approximation**, and **Deep Q-Networks (DQN)** in a text-based interactive environment.

The project demonstrates how reinforcement learning methods can evolve from explicit state-action value tables to learned function approximators as the state space becomes more complex.

## Overview

The agent interacts with a text-based **Home World** environment. At each step, the environment provides textual descriptions of the agent's current room and quest.

The agent must learn to select an appropriate combination of:

- **Action** — e.g. `eat`, `sleep`, `watch`, `exercise`, or `go`
- **Object** — e.g. `apple`, `bed`, `tv`, `bike`, or a navigation direction

The objective is to complete the assigned quest while maximizing cumulative discounted reward.

Examples of quests include:

- "You are bored."
- "You are getting fat."
- "You are hungry."
- "You are sleepy."

The environment contains four locations:

- Living Room
- Garden
- Kitchen
- Bedroom

Each location can have multiple textual descriptions, requiring the agent to reason from observed text rather than directly accessing the underlying environment state.

---

## Methods

### 1. Tabular Q-Learning

The first implementation represents each unique textual state using a discrete index.

A state consists of:

`state = (room_description, quest_description)`

where:

- `room_description` represents the textual description of the agent's current room;
- `quest_description` represents the agent's current quest.

The agent maintains a Q-table containing a Q-value for each state-command pair.

Each command consists of an action and an object:

`command = (action, object)`

For example:

- `eat apple`
- `sleep bed`
- `watch tv`
- `exercise bike`
- `go north`

For each transition `(s, c, r, s')`, the Q-value is updated using the Q-learning rule:

`Q(s, c) = Q(s, c) + alpha * [r + gamma * max Q(s', c') - Q(s, c)]`

where:

- `alpha` is the learning rate;
- `gamma` is the discount factor;
- `r` is the immediate reward;
- `max Q(s', c')` represents the highest Q-value among all possible commands in the next state.

For terminal states, there is no future Q-value, so the update becomes:

`Q(s, c) = Q(s, c) + alpha * [r - Q(s, c)]`

An **epsilon-greedy policy** is used to balance exploration and exploitation:

- with probability `epsilon`, the agent selects a random command;
- with probability `1 - epsilon`, the agent selects the command with the highest current Q-value.

Experiments were performed with different values of `epsilon` and `alpha` to investigate how exploration and learning rate affect convergence and agent performance.

---

### 2. Linear Function Approximation

Maintaining an explicit Q-value for every state-action pair becomes impractical as the state space grows.

To address this limitation, the second implementation approximates the Q-function using a linear model:

`Q(s, c; theta) = phi(s, c)^T * theta`

where:

- `phi(s, c)` is the feature vector representing the state-command pair;
- `theta` is the learnable parameter vector.

Textual states are first converted into vector representations using a **Bag-of-Words (BoW)** representation:

`Text State → Bag-of-Words → State Feature Vector`

To distinguish different commands, the state feature vector is placed into a command-specific feature block:

`phi(s, c) = [0, ..., psi_R(s), ..., 0]`

This allows each command to learn its own parameter vector while using the same state representation framework.

For each transition `(s, c, r, s')`, the temporal-difference (TD) error is:

`delta = r + gamma * max Q(s', c') - Q(s, c)`

where the maximum is taken over all possible next commands `c'`.

For terminal states, there is no future Q-value, so:

`delta = r - Q(s, c)`

The parameters are updated using:

`theta = theta + alpha * delta * phi(s, c)`

where `alpha` is the learning rate.

This experiment also demonstrates an important limitation of linear function approximation: a simple linear model may not have sufficient representational capacity to accurately approximate the Q-function, even in a relatively small environment.

---

### 3. Deep Q-Network (DQN)

The final implementation replaces the linear Q-function approximator with a **Deep Q-Network (DQN)**.

The textual state is first transformed into a Bag-of-Words feature vector:

`Text State → Bag-of-Words → DQN → Q-values`

The neural network predicts Q-values for possible actions and objects.

For each transition:

`(s, a, r, s')`

the Q-learning target for a non-terminal state is:

`y = r + gamma * max Q(s', a')`

where the maximum is taken over all possible next actions `a'`.

For a terminal state, there is no future reward, so the target becomes:

`y = r`

The prediction error is measured using the squared loss:

`L = 1/2 * (y - Q(s, a))^2`

The DQN parameters are then updated using gradient-based optimization to minimize this loss.

An **epsilon-greedy policy** is used for action selection:

- with probability `epsilon`, the agent selects a random action;
- with probability `1 - epsilon`, the agent selects the action with the highest predicted Q-value.

During training, the DQN parameters are updated from observed transitions. During testing, the learned policy is evaluated without updating the network.

---

## Environment

The Home World contains four rooms with multiple natural-language descriptions.

| Room | Example Relevant Interaction |
|------|------------------------------|
| Living Room | Watch TV |
| Garden | Exercise / ride bike |
| Kitchen | Eat an apple |
| Bedroom | Sleep in bed |

The environment provides positive reward for successfully completing a quest and negative rewards for unnecessary or invalid actions.

Each episode terminates when either:

1. the agent successfully completes the quest, or
2. the maximum number of environment steps is reached.

---

## Evaluation

The project evaluates and compares three Q-learning approaches:

1. **Tabular Q-Learning**
2. **Q-Learning with Linear Function Approximation**
3. **Deep Q-Learning (DQN)**

All three approaches are evaluated using **cumulative discounted episodic reward**:

`G = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(T-1) * r_(T-1)`

Training and testing are performed separately.

During training, each agent updates its Q-function according to its representation:

- **Tabular Q-Learning:** directly updates entries in a Q-table.
- **Linear Q-Learning:** updates the parameter matrix `theta` using gradient-based updates.
- **Deep Q-Learning:** updates the weights of a neural network using the temporal-difference target and gradient descent.

During testing, the learned policy is evaluated without updating the Q-function or model parameters.

Performance is averaged across multiple testing episodes and experimental runs to reduce the effect of randomness and provide a more reliable estimate of agent performance.

The experiments demonstrate how the choice of Q-function representation affects learning behavior and performance, progressing from discrete tabular representations to linear function approximation and finally to neural-network-based approximation.

---

## Experimental Progression

The project illustrates the progression from exact value storage to learned representations:

| Method | State Representation | Q Representation | Main Limitation |
|--------|----------------------|------------------|-----------------|
| Tabular Q-Learning | Discrete state index | Q-table | Does not scale to large state spaces |
| Linear Q-Learning | Bag-of-Words | Linear function | Limited representational capacity |
| DQN | Bag-of-Words | Neural network | More computationally expensive |

This progression demonstrates why function approximation becomes important when reinforcement learning agents operate in increasingly large or complex observation spaces.

---

## Key Concepts Implemented

- Q-Learning
- Bellman updates
- Temporal-difference learning
- Epsilon-greedy exploration
- Exploration vs. exploitation
- Cumulative discounted rewards
- Bag-of-Words state representations
- Feature engineering
- Linear function approximation
- Gradient-based Q-learning
- Deep Q-Networks
- PyTorch model training
- Reinforcement learning evaluation

---

## Tech Stack

- Python
- NumPy
- PyTorch
- Matplotlib

---

## Key Takeaways

This project highlights the trade-off between model simplicity and representational power in reinforcement learning.

**Tabular Q-Learning** performs well when the complete state-action space is small enough to enumerate, but its memory requirements grow rapidly with the number of states.

**Linear function approximation** provides a scalable alternative by sharing parameters across states, but its performance depends heavily on the quality and expressiveness of the chosen features.

**Deep Q-Learning** provides greater representational capacity by learning nonlinear relationships between textual state features and expected future rewards.

Overall, the project demonstrates a natural progression:

**Q-Table → Feature Engineering → Linear Approximation → Neural Q-Function**

and provides practical experience with both classical and deep reinforcement learning techniques.