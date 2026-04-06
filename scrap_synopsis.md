# A Rational AI Agent for City Decision-Making
**PAI Lab Assignment**  
**Name:** Divya Battu (230968042), M Sarayu (230968052)

## Introduction

Urban environments are complex, constantly changing systems that involve transportation networks, varying population densities, infrastructure constraints, and challenges related to efficient resource allocation. Decision-making in such environments often involves optimising multiple conflicting objectives, such as minimising travel time, ensuring equitable resource distribution, reducing congestion, and maintaining service accessibility.

Traditional urban planning methods rely heavily on statistical analysis and manual heuristics. However, these methods lack adaptability and explainability in dynamic, constraint-heavy environments. Artificial Intelligence, particularly classical AI approaches such as rational agents, search algorithms, constraint satisfaction problems (CSPs), adversarial reasoning, and reinforcement learning, offers a structured framework for modelling and solving decision-making problems.

This project proposes the development of an Explainable Rational AI Agent for Urban Decision-Making. The system will simulate a city environment using real-world maps and traffic datasets and apply classical AI algorithms to optimise decisions such as facility placement and traffic routing. Unlike purely data-driven machine learning systems, the proposed agent will make decisions using formal search, reasoning, and optimisation methods and provide transparent explanations for its actions.

The project integrates multiple core concepts from the Principles of Artificial Intelligence syllabus into a unified real-world system.

## Problem Statement

Modern cities face challenges such as:
1. Public facilities (emergency rooms, charging stations, and hospitals) are not positioned optimally.
2. Traffic jams brought on by ineffective routing.
3. Inequality of resources among urban areas.
4. Automated decision systems' inability to be explained.

The problem addressed in this project is to design and implement a rational AI agent capable of making optimal, explainable decisions for urban planning tasks, such as facility placement and traffic routing, under real-world constraints.

In particular, the system needs to:
1. Model a city environment as a graph.
2. Use traditional search algorithms to optimise routing choices.
3. Address the facility-allocation constraint-satisfaction problem.
4. Model congestion or conflicting goals using adversarial reasoning.
5. Adapt to dynamic environments using reinforcement learning.
6. Give clear justification for every choice made.

The core challenge lies in integrating multiple AI paradigms into a unified, interpretable decision-making framework operating on realistic datasets.

## Literature Review

The concept of rational agents underpins classical Artificial Intelligence. A rational agent perceives its environment and selects actions that maximise a predefined performance measure. This framework provides a structured approach to modelling intelligent decision-making systems.

Search algorithms are fundamental to problem-solving in AI. Uninformed methods, such as Breadth-First Search (BFS) and Uniform Cost Search (UCS), systematically explore the state space, while informed algorithms, such as A*, use heuristics to improve efficiency. These techniques are widely used in navigation and route optimisation problems.

Constraint Satisfaction Problems (CSPs) model problems with multiple constraints that must be satisfied simultaneously. Techniques such as backtracking and constraint propagation improve the efficiency of solving combinatorial problems, making CSPs suitable for facility placement and resource allocation tasks.

These established AI methodologies provide the theoretical basis for developing an explainable urban decision-making system in this project.

## Objectives

The primary objective of this project is to develop an AI system that applies classical AI techniques to real-world urban decision problems.

1. To design a rational agent using the PEAS formulation for urban decision-making.
2. To implement graph-based search algorithms (BFS, UCS, A*) for optimal routing.
3. To model facility placement as a Constraint Satisfaction Problem.
4. To provide explainable outputs for every decision.
5. To develop an interactive visualisation interface for real-time demonstration.

## References

1. [AI Agents in Urban Planning: A New Paradigm for Urban Simulation](https://medium.com/urban-ai/ai-agents-in-urban-planning-a-new-paradigm-for-urban-simulation-41c4c210e4a8)
2. [Research Document: arXiv:2507.14730v2](https://www.arxiv.org/pdf/2507.14730v2)
3. [Simulating Cities with AI Agents - The Alan Turing Institute](https://www.turing.ac.uk/research/research-projects/simulating-cities-ai-agents)