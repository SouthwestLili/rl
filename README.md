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

\[
s = (s_r, s_q)
\]

where:

- \(s_r\) represents the room description
- \(s_q\) represents the quest description

The agent maintains a Q-table over state-action pairs and updates its values using:

\[
Q(s,a) \leftarrow Q(s,a)
+ \alpha
\left[
r + \gamma \max_{a'} Q(s',a')
- Q(s,a)
\right]
\]

An **epsilon-greedy policy** balances exploration and exploitation:

- with probability \(\epsilon\), select a random action;
- with probability \(1-\epsilon\), select the action with the highest current Q-value.

Experiments were performed to investigate how the exploration parameter \(\epsilon\) and learning rate \(\alpha\) affect convergence.

---

### 2. Linear Function Approximation

Maintaining an explicit Q-value for every state-action pair becomes impractical as the state space grows.

To address this limitation, the second implementation approximates the Q-function using a linear model:

\[
Q(s,c;\theta) = \phi(s,c)^T\theta
\]

where:

\[
- \(\phi(s,c)\) is the feature representation of a state-command pair;
- \(\theta\) contains the learnable parameters.
\]

Textual states are converted into vector representations using a **Bag-of-Words (BoW)** representation.

Action-dependent feature blocks allow different commands to learn separate parameter vectors while sharing the same state representation framework.

The parameters are updated using the temporal-difference error:

\[
\delta =
r + \gamma \max_{c'}Q(s',c';\theta)
- Q(s,c;\theta)
\]

and a gradient update:

\[
\theta \leftarrow
\theta + \alpha \delta \phi(s,c)
\]

This experiment also demonstrates an important limitation: a simple linear model may not have sufficient representational capacity to accurately approximate the Q-function even in a relatively small environment.

---

### 3. Deep Q-Network

The final implementation replaces the linear Q-function approximator with a **Deep Q-Network (DQN)**.

The textual state is first transformed into a Bag-of-Words feature vector:

\[
s_{\text{text}}
\rightarrow
\text{Bag-of-Words}
\rightarrow
\text{DQN}
\rightarrow
Q\text{-values}
\]

The neural network predicts Q-values for possible actions and objects.

For a transition:

\[
(s,a,r,s')
\]

the Q-learning target is based on:

\[
y =
r + \gamma \max_{a'} Q(s',a')
\]

for non-terminal states, while terminal transitions use the immediate reward as the target.

The model is optimized by minimizing the difference between the predicted Q-value and the Q-learning target using gradient-based optimization.

An epsilon-greedy policy is again used during both training and evaluation.

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

Agent performance is measured using **cumulative discounted episodic reward**:

\[
G =
\sum_{t=0}^{T-1}
\gamma^t r_t
\]

Training and testing are performed separately.

During training, the agent updates its Q-values or model parameters using collected transitions.

During testing, the learned policy is evaluated without updating the model. Performance is averaged across multiple episodes and experimental runs to reduce the effect of randomness.

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