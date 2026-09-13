# Operations Research & Optimization Problems

This repository contains Python implementations of two Operations Research optimization problems completed as part of an academic assignment.

The objective is to understand how optimization techniques can be formulated, implemented, and used to obtain optimal solutions programmatically.

## Problems Covered

### 1. Linear Programming Problem

A Linear Programming Problem (LPP) is formulated and solved using optimization techniques to determine the optimal values of the decision variables while satisfying the given constraints.

**Key concepts covered:**
- Decision variables
- Objective function
- Constraints
- Feasible region
- Optimal solution
- Maximum/Minimum objective value

### 2. Transportation Problem

The Transportation Problem determines the optimal allocation of goods from multiple sources to multiple destinations while minimizing the total transportation cost.

The problem is solved using:

- **Vogel's Approximation Method (VAM)** – to obtain an initial basic feasible solution
- **MODI (Modified Distribution) Method** – to test optimality and improve the solution iteratively

**Key concepts covered:**
- Supply and demand
- Transportation cost matrix
- Balanced transportation problem
- Initial Basic Feasible Solution (IBFS)
- Vogel's Approximation Method
- MODI method
- Opportunity cost
- Optimal transportation plan
- Minimum transportation cost

## Technologies Used

- Python 3
- NumPy
- Jupyter Notebook / Python
- Git & GitHub

## Repository Structure

```text
Optimization-Problems/
│
├── Linear_Programming/
│   ├── lpp.py
│   └── README.md
│
├── Transportation_Problem/
│   ├── transportation.py
│   └── README.md
│
├── requirements.txt
└── README.md
