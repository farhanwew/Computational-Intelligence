# Muhammad Farhan Arya Wicaksono (5054231011)

# Logical Agents

It’s how AI approaches intelligence.
Knowledge-based agents can:

- accept new tasks in the form of explicitly described goals
- achieve competence quickly by being told or learning new knowledge about the environment
- adapt to changes in the environment by updating the relevant knowledge

```python
from logic import *
from utils import *
from notebook import psource
import pandas as pd
```

Create a program for the Wumpus World problem based on the book **Artificial Intelligence: A Modern Approach**, page 238.

![image.png](image.png)

- Create several propositions (rules) R1 to Rn in sequence based on the Wumpus World rules, starting from the agent’s coordinate [1,1] until the agent successfully retrieves the gold at coordinate [2,3] safely.
- Perform inference using **TT-entails** (for students with an odd NRP).
- Perform inference using **Resolution** (for students with an even NRP).

In the report, write the program code and analyze the program, especially explaining why the result is such by using mathematical logic.

---

## The Wumpus World

- A cave with many rooms connected by corridors.
- **Wumpus**: A beast that eats anyone who enters its room.
- The Wumpus can be killed by the agent, but the agent only has one arrow.
- Some rooms have pits where once entered, you cannot exit.
- Goal = find the gold.

### The Wumpus World: PEAS definition

---

**Performance Measure**

- +1000: for retrieving gold
- -1000: for falling into a pit or being eaten by the Wumpus
- -1: for each action taken
- -10: for using the arrow
- The game ends when the agent dies or climbs out of the cave.

**Environment**

- A 4 × 4 grid of rooms.
- The agent always starts in the square labeled [1,1], facing to the right.
- The locations of the gold and the Wumpus are randomly chosen from the squares other than the start square.
- Each square except the start can be a pit with a probability of 0.2.

**Actuators**

- The agent can move Forward, TurnLeft by 90°, or TurnRight by 90°.
- The agent dies if it enters a square containing a pit or a live Wumpus.
- If an agent tries to move forward and bumps into a wall, it doesn’t move.
- The action **Grab** is used to pick up the gold if it’s in the same square as the agent.
- The action **Shoot** fires an arrow in a straight line in the direction the agent is facing.
- The action **Climb** can be used to climb out of the cave, but only from square [1,1].

**Sensors**

- In the square containing the Wumpus or adjacent squares (not diagonally), the agent perceives a **Stench**.
- In squares adjacent to a pit, the agent perceives a **Breeze**.
- In the square with the gold, the agent perceives a **Glitter**.
- When the agent walks into a wall, it perceives a **Bump**.
- When the Wumpus is killed, it emits a **Scream** that can be heard anywhere in the cave.

---

## Inference in Propositional Knowledge Base

- Enumerating all possible models where the Knowledge Base (KB) is true.
- Checking if the given query is also true in those models.

We enumerate all symbols $n$ in `KB` and calculate $2^n$ models, checking if `KB` and $lpha$ hold in these models.

# Code

The following program is based on references from Aiman Python with some additional changes:

```python
def variables(s):
    """Return a set of the variables in expression s.
    >>> variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)')) == {x, y, z}
    True
    """
    return {x for x in subexpressions(s) if is_variable(x)}
```

```python
# Returns True if the propositional logic expression is true in the model,
# False if false. If the model doesn’t assign values to all propositions,
# the function may return None, indicating 'not clear'; this can happen
# even when the expression is a tautology.

def pl_true(exp, model={}):
    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':
        return pt != qt
    else:
        raise ValueError('Illegal operator in logic expression' + str(exp))
```

```python
def tt_check_all_edit(kb, alpha, symbols, model, results_list):
    if not symbols:
        if pl_true(kb, model):
            result = pl_true(alpha, model)
            assert result in (True, False)
            model['kb'] = True
            results_list.append(model)
            return result
        else:
            result_info = f"KB false, Model: {model}"
            model['kb'] = False
            results_list.append(model)
            return True
    else:
        P, rest = symbols[0], symbols[1:]
        true_result = tt_check_all_edit(kb, alpha, rest, extend(model, P, True), results_list)
        false_result = tt_check_all_edit(kb, alpha, rest, extend(model, P, False), results_list)
        return (true_result and false_result)
```

```python
def tt_entails_edit(kb, alpha):
    assert not variables(alpha)
    symbols = list(prop_symbols(kb & alpha))
  
    print(f"Symbols: {symbols}")
    print(f"KB: {kb}
")
    print(f"Alpha: {alpha}")
    print('-' * 20)
  
    results_list = []
    result = tt_check_all_edit(kb, alpha, symbols, {}, results_list)
  
    print(f"Final result: {result}")

    return result, results_list
```

---

This algorithm checks the truth of each row in the truth table for the expression KB ⟹ α.

The algorithm generates all possible truth value combinations for the symbols. For each combination (model), it checks whether that model is consistent with the KB. If consistent, it checks whether the query (αlpha) is true in that model.

The function `tt_entails()` retrieves the symbols from the query and invokes `tt_check_all()` with appropriate parameters.

## Steps

### KB is true if $R_1$ to $R_n$ are true:

- $P_{x,y}$ is true if there is a pit in [x, y].
- $W_{x,y}$ is true if there is a Wumpus in [x, y], dead or alive.
- $B_{x,y}$ is true if the agent perceives a breeze in [x, y].
- $S_{x,y}$ is true if the agent perceives a stench in [x, y].
- $G_{x,y}$ is true if the agent perceives gold in [x, y].


|   | 1      | 2  | 3 | 4 | 5 |
| --- | -------- | ---- | --- | --- | --- |
|   |        |    |   |   |   |
| 4 |        |    |   |   |   |
| 3 |        |    |   |   |   |
| 2 | ok     |    |   |   |   |
| 1 | (A,ok) | ok |   |   |   |

[1,1] is safe; there’s nothing.

---

The remaining steps follow from this logic...
