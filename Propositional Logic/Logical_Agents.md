# Logical Agents

* representasi dari complex world
* Melakukan inferensi dari representasi menjadi representasi baru
* Membuat kesimpulan aksi berdasarkan representasi representasi baru

# Knowledge-based Agents

* accept new tasks
* achive competence
* adapt to changes

# Component

A knowledge base: a set of sentences.
Each sentence is expressed in a language: a knowledge representation language.

To add new sentences to the knowledge base and to query what is known:
◦ Operations are TELL and ASK, respectively.
◦ Both operations may involve inference—that is, deriving new sentences from old




## The Wumpus World
- Goa yang memiliki banyak ruangan yang mana terhubung dengan lorong lorong
- Wumpus: Binatang buas yang memakan siapa pun yang masuk kedalam ruangannya
- wumpus dapat dibunuh oleh agent, tetapi agen hanya memiliki satu panah.
- Beberapa ruangan memeliki jurang yang mana jika sudah masuk tidak bisa keluar
- Goal = mencari gold

The Wumpus World: PEAS definition
---
Performance measure
- +1000: gold
- -1000: fall into a pit or being eaten by the W umpus
- -1: for each action taken
- -10: for using up the arrow
- The game ends either when the agent dies or when the agent climbs out of the cave.
---
Environment
- A 4 × 4 grid of rooms.
- The agent always starts in the square labeled [1,1], facing to the right.
- The locations of the gold and the wumpus are chosen randomly, with a uniform distribution, from the squares other than the start square.
- Each square other than the start can be a pit, with probability 0.2.
---
Actuators
- The agent can move Forward, TurnLeft by 90◦, or TurnRight by 90◦.
- The agent dies if it enters a square containing a pit or a live wumpus.
- If an agent tries to move forward and bumps into a wall, then the agent does not move.
- The action Grab is used to pick up the gold if it is in the same square as the agent.
- The action Shoot is used to fire an arrow in a straight line in the direction the agent is facing.
- The action Climb can be used to climb out of the cave, but only from square [1,1]
---
Sensors
- In the square containing the wumpus and in the directly (not diagonally) adjacent squares, the agent will perceive a Stench.
- In the squares directly adjacent to a pit, the agent will perceive a Breeze.
- In the square where the gold is, the agent will perceive a Glitter.
- When an agent walks into a wall, it will perceive a Bump.
- When the wumpus is killed, it emits a Scream that can be perceived anywhere in the cave.
- Percept example: [Stench, Breeze, None, None, None]

## Inference in Propositional Knowledge Base
- menghitung semua model yang mungkin di mana KB adalah benar  
- memeriksa apakah juga benar dalam model-model ini. 

membuat daftar simbol $n$ dalam `KB` dan menghitung model $2^{n}$ dengan cara yang lebih mendalam dan memeriksa kebenaran `KB` dan $\alpha$.