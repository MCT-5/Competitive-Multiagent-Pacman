# Competitive Multi-Agent Pacman: Search, Auctions and Decision Trees

A multi-agent AI project in which several Pacman agents share one maze, flee from ghosts, collect coins and **compete for scarce resources sold at auction**. Each agent combines three classic AI techniques:

| Module | Topic | Role in the agent |
|---|---|---|
| **A** | Search and adversarial search (BFS, A\*, Minimax with Alpha-Beta pruning) | Low-level navigation and immediate survival |
| **B** | Auctions and resource allocation (Vickrey and Dutch) | Competitive economy and budget management |
| **C** | Decision trees (ID3, implemented from scratch) | High-level decision-making: when and how to bid |

The project is written in pure Python 3.10+. Only the optional parts need extra packages (`matplotlib` for charts, `tkinter` for the GUI).

> **Diagrams.** The figures below are written in [Mermaid](https://mermaid.js.org), which GitHub, GitLab, VS Code, Typora and PyCharm (Markdown preview) render automatically. PNG and SVG copies for slides or reports are in the `diagrams/` folder.

---

## 1. Game concept

Up to four Pacmans play in the same 21×11 maze together with three ghosts. The coins they collect are their **capital**. Special resources are not lying around in the maze. They are **sold at auction every 50 turns**:

| Item | Category | Base value | Effect |
|---|---|---|---|
| Power Pellet | Points | 25 | 25 turns in which the agent can eat ghosts (+20 coins each) |
| Extra Life | Survival | 30 | +1 life |
| Shield | Survival | 18 | 20 turns of immunity to ghosts |
| Radar | Points | 10 | 40 turns of full map knowledge |

Without a radar, an agent only sees coins within a radius of 5 cells. Ghosts are always visible.

**Winning an item is not enough.** The item appears at a random location in the maze (4 to 12 steps away), and only the winner can pick it up, so the agent has to navigate there.

**Final score** = remaining wallet + 0.3 points per turn survived + 10 points per remaining life.

This creates the central tension of the project: spending coins on survival tools now versus keeping them for the final score.

### Ghost behaviour

- With probability 0.7 a ghost chases the nearest Pacman that has no active Power Pellet. Otherwise it moves randomly.
- Ghosts flee from a Pacman with an active Power Pellet that is within 4 cells.
- A newly spawned or respawned Pacman has 3 turns of grace shielding.
- Coins respawn at random with probability 0.25 per turn, so the economy never dries up.

### Turn loop

Every turn of the engine follows the same sequence.

```mermaid
flowchart TD
    T0(["Start of turn t"])
    Q0{"Auction turn?<br/>(every 50 turns)"}
    AU["Auction<br/>all living agents bid"]
    AC["Each living Pacman<br/>(random order) perceives<br/>and chooses a move"]
    MV["Move validated<br/>(1 step, no walls)<br/>collect coin / item"]
    C1["Resolve collisions<br/>shield: safe<br/>power: eat ghost<br/>otherwise: lose a life"]
    GH["Ghosts move<br/>(chase or flee)"]
    C2["Resolve collisions<br/>again"]
    UP["Update timers and<br/>survival counter<br/>maybe respawn a coin"]
    Q1{"Game over?<br/>(turn limit or<br/>all Pacmans dead)"}
    SC(["Final score =<br/>wallet + survival + lives"])
    T0 --> Q0
    Q0 -->|"yes"| AU
    Q0 -->|"no"| AC
    AU --> AC
    AC --> MV
    MV --> C1
    C1 --> GH
    GH --> C2
    C2 --> UP
    UP --> Q1
    Q1 -->|"no: next turn"| T0
    Q1 -->|"yes"| SC
    classDef core fill:#eceef4,stroke:#8a90a6,color:#222,stroke-width:1.5px
    classDef mA fill:#cfe3ff,stroke:#4a86d6,color:#222,stroke-width:1.5px
    classDef mB fill:#ffe0b8,stroke:#e08a1e,color:#222,stroke-width:1.5px
    classDef term fill:#dddddd,stroke:#777777,color:#222,stroke-width:1.5px
    classDef test fill:#f2f2f7,stroke:#8a90a6,color:#222,stroke-width:1.5px
    class MV,C1,GH,C2,UP core
    class AC mA
    class AU mB
    class T0,SC term
    class Q0,Q1 test
```

*Figure 1. Game turn loop.* (image version: `diagrams/01_turn_loop.png`)

---

## 2. Architecture

```
pacman_mad/
├── search.py          Module A: BFS, A*, Minimax + Alpha-Beta
├── auction.py         Module B: Vickrey and Dutch auctioneer, item catalogue
├── decision_tree.py   Module C: ID3 from scratch, log generation, persistence
├── agents.py          IntelligentAgent (integrates A + B + C), random agent
├── game.py            Game engine: maze, ghosts, items, auctions, scoring
├── personalities.py   Four decision-tree "personalities"
├── metrics.py         Game recorder, JSON snapshots, Pearson correlation
├── experiments.py     Three experiments + charts + CSV export
├── gui.py             Tkinter GUI with live auction and decision-tree panel
├── main.py            Basic tournament and ASCII rendering
├── train_tree.py      Trains and evaluates the base decision tree
└── tests/             142 unit tests (unittest)
```

```mermaid
flowchart TD
    GAME["game.py<br/>game engine"]
    AG["agents.py<br/>IntelligentAgent"]
    PERS["personalities.py<br/>4 decision trees"]
    MET["metrics.py<br/>recorder + snapshots"]
    LOG["data/logs.csv<br/>models/tree.json"]
    subgraph ENTRY["Entry points"]
        MAIN["main.py<br/>basic tournament"]
        GUI["gui.py<br/>Tkinter GUI"]
        EXP["experiments.py<br/>charts + CSV"]
        TRN["train_tree.py<br/>train + evaluate"]
    end
    subgraph MODS["AI modules"]
        MA["search.py<br/>Module A<br/>BFS, A*, Minimax + Alpha-Beta"]
        MB["auction.py<br/>Module B<br/>Vickrey / Dutch auctioneer"]
        MC["decision_tree.py<br/>Module C<br/>ID3 from scratch"]
    end
    MAIN --> GAME
    GUI -->|"live game"| GAME
    GUI -->|"snapshots / replay"| MET
    EXP -->|"rosters"| PERS
    EXP -->|"records games"| MET
    MET -->|"wraps"| GAME
    PERS -->|"builds agents"| AG
    PERS -->|"trains trees"| MC
    GAME -->|"act() / bid()"| AG
    GAME -->|"run auction"| MB
    AG -->|"navigation"| MA
    AG -->|"classify state"| MC
    TRN -->|"writes"| LOG
    LOG -->|"loaded by"| MC
    classDef core fill:#eceef4,stroke:#8a90a6,color:#222,stroke-width:1.5px
    classDef data fill:#fff6c8,stroke:#c9a800,color:#222,stroke-width:1.5px
    classDef mA fill:#cfe3ff,stroke:#4a86d6,color:#222,stroke-width:1.5px
    classDef mB fill:#ffe0b8,stroke:#e08a1e,color:#222,stroke-width:1.5px
    classDef mC fill:#d3f2d0,stroke:#4fa84a,color:#222,stroke-width:1.5px
    classDef ui fill:#e6e0fa,stroke:#7b68c8,color:#222,stroke-width:1.5px
    class GAME,AG,PERS,MET core
    class LOG data
    class MA mA
    class MB mB
    class MC mC
    class MAIN,GUI,EXP,TRN ui
```

*Figure 2. System architecture: entry points, core files and the three AI modules.* (image version: `diagrams/02_architecture.png`)

### Module A: navigation and search (`search.py`)

- **BFS** finds the nearest coin or the nearest unexplored cell, optionally avoiding cells next to ghosts.
- **A\*** (Manhattan heuristic) walks to won items. A danger map adds extra cost to cells near ghosts, so paths take detours around threats.
- **Minimax with Alpha-Beta pruning** handles escape. Pacman is the MAX player and the nearest ghosts are adversarial MIN players. The evaluation function rewards distance from ghosts and open escape routes. It also gives a small pull towards the current goal, which only matters when ghosts are far away.

### Module B: auctions (`auction.py`)

An `Auctioneer` collects one sealed bid per agent and decides the winner and the price:

- **Vickrey (second price):** the highest bid wins and pays the second-highest bid. With a single bidder the price is the reserve price.
- **Dutch:** the price starts high and descends. The first agent whose maximum price is reached wins and pays the current price.

Ties are broken randomly. Bids are always clipped to the bidder's wallet, so a wallet can never go negative.

#### Auction flow

From the agent's bid to the winner picking up the item:

```mermaid
flowchart TD
    S0(["Auction turn<br/>item drawn at random"])
    GO(["Winner walks there with A*<br/>and gets the effect"])
    subgraph PER["Per agent"]
        FE["Build features<br/>ghost distance, life,<br/>coins, item type"]
        TR["ID3 returns<br/>class A / B / C"]
        CQ{"Class?"}
        BA["A: aggressive<br/>bid = value x mult_a"]
        BB["B: conservative<br/>bid = min(value x mult_b,<br/>half of wallet)"]
        BC["C: ignore auction<br/>bid = 0"]
        CL["Clip bid<br/>to wallet"]
    end
    subgraph AUC["Auctioneer"]
        CO["Auctioneer<br/>collects all bids"]
        TY{"Auction type?"}
        VI["Vickrey<br/>highest bid wins and pays<br/>the second-highest bid<br/>(reserve if alone)"]
        DU["Dutch<br/>price falls from 60 in steps of 2;<br/>first agent whose max price<br/>is reached wins"]
        WI["Winner pays;<br/>item appears 4 to 12<br/>steps away"]
    end
    S0 --> FE
    FE --> TR
    TR --> CQ
    CQ -->|"A"| BA
    CQ -->|"B"| BB
    CQ -->|"C"| BC
    BA --> CL
    BB --> CL
    BC --> CL
    CL --> CO
    CO --> TY
    TY -->|"Vickrey"| VI
    TY -->|"Dutch"| DU
    VI --> WI
    DU --> WI
    WI --> GO
    classDef core fill:#eceef4,stroke:#8a90a6,color:#222,stroke-width:1.5px
    classDef leafA fill:#f6b9a6,stroke:#e4572e,color:#222,stroke-width:1.5px
    classDef leafB fill:#fbe0a0,stroke:#f3a712,color:#222,stroke-width:1.5px
    classDef leafC fill:#a9e3df,stroke:#29a19c,color:#222,stroke-width:1.5px
    classDef mB fill:#ffe0b8,stroke:#e08a1e,color:#222,stroke-width:1.5px
    classDef mC fill:#d3f2d0,stroke:#4fa84a,color:#222,stroke-width:1.5px
    classDef term fill:#dddddd,stroke:#777777,color:#222,stroke-width:1.5px
    classDef test fill:#f2f2f7,stroke:#8a90a6,color:#222,stroke-width:1.5px
    class FE,CL,WI core
    class BA leafA
    class BB leafB
    class BC leafC
    class CO,VI,DU mB
    class TR mC
    class S0,GO term
    class CQ,TY test
```

*Figure 3. How an auction is resolved.* (image version: `diagrams/03_auction_flow.png`)

### Module C: decision tree (`decision_tree.py`)

A hand-written **ID3** classifier (entropy and information gain, categorical features, optional depth limit, JSON save/load). It classifies the game state using four features:

| Feature | Values |
|---|---|
| `dist_fantasmes` (ghost distance) | A prop (≤ 4 cells) / Lluny |
| `vida` (life) | Baixa (≤ 1 life) / Alta |
| `monedes` (coins) | Pobres / Rics (above a per-agent threshold) |
| `tipus_item` (item type) | Supervivencia / Punts |

and returns one of three strategies:

| Class | Strategy | Bid |
|---|---|---|
| **A** | Bid aggressively | `value × mult_a` |
| **B** | Bid conservatively | `min(value × mult_b, 50% of wallet)` |
| **C** | Ignore the auction and flee | 0 |

#### Example tree

The trained *Equilibrat* tree, with redundant splits merged (the decisions are identical to the full ID3 tree):

```mermaid
flowchart TD
    R{"Ghost<br/>distance?"}
    N1{"Item<br/>type?"}
    N2{"Life?"}
    N3{"Coins?"}
    F1{"Item<br/>type?"}
    F2{"Coins?"}
    F3{"Life?"}
    L1["C<br/>ignore + flee"]
    L2["C<br/>ignore + flee"]
    L3["C<br/>ignore + flee"]
    L4["A<br/>bid aggressively"]
    L5["C<br/>ignore + flee"]
    L6["B<br/>bid carefully"]
    L7["B<br/>bid carefully"]
    L8["A<br/>bid aggressively"]
    R -->|"near (4 or less)"| N1
    R -->|"far"| F1
    N1 -->|"points"| L1
    N1 -->|"survival"| N2
    N2 -->|"high"| L2
    N2 -->|"low"| N3
    N3 -->|"poor"| L3
    N3 -->|"rich"| L4
    F1 -->|"points"| F2
    F1 -->|"survival"| F3
    F2 -->|"poor"| L5
    F2 -->|"rich"| L6
    F3 -->|"high"| L7
    F3 -->|"low"| L8
    classDef leafA fill:#f6b9a6,stroke:#e4572e,color:#222,stroke-width:1.5px
    classDef leafB fill:#fbe0a0,stroke:#f3a712,color:#222,stroke-width:1.5px
    classDef leafC fill:#a9e3df,stroke:#29a19c,color:#222,stroke-width:1.5px
    classDef test fill:#f2f2f7,stroke:#8a90a6,color:#222,stroke-width:1.5px
    class L4,L8 leafA
    class L6,L7 leafB
    class L1,L2,L3,L5 leafC
    class R,N1,N2,N3,F1,F2,F3 test
```

*Figure 4. Decision tree of the Equilibrat personality.* (image version: `diagrams/04_decision_tree.png`)

---

## 3. Agent life cycle

On every turn, an `IntelligentAgent` does the following:

1. **Perceive** its position, the ghosts, the visible coins, its pending items and its resources.
2. **Classify the state** with the decision tree. When no item is on sale, it uses the item category the agent would currently need: *survival* if it has one life or a ghost is near, *points* otherwise.
3. **Negotiate** (only on auction turns): estimate the item's value from the context, then bid according to the class the tree returned.
4. **Act**, in priority order:
   1. A ghost is too close and the agent is vulnerable: **Minimax / Alpha-Beta** flight. The trigger distance is wider (fewer than 5 cells) for class C and 3 otherwise.
   2. A Power Pellet is active: hunt the nearest ghost with **A\***.
   3. There is a won item waiting: go to it with **A\***.
   4. Otherwise: **BFS** to the nearest visible coin, or explore unvisited cells.

```mermaid
flowchart TD
    P(["Perceive view<br/>position, ghosts, visible coins,<br/>items, wallet, lives"])
    F["Ghost distance (BFS)<br/>build 4 features"]
    CL["Classify state<br/>with ID3: class A / B / C"]
    Q1{"Vulnerable (no shield/power)<br/>and ghost closer than 3?<br/>(5 if class C)"}
    M["Minimax + Alpha-Beta<br/>flee"]
    Q2{"Power Pellet active<br/>and ghost nearby?"}
    H["A* hunt<br/>nearest ghost"]
    Q3{"Won item<br/>pending?"}
    I["A* to the item<br/>avoiding danger zones"]
    Q4{"Visible coin?"}
    BC["BFS to nearest coin<br/>(avoid cells next to ghosts)"]
    EX["BFS explore<br/>nearest unvisited cell"]
    MV(["Move"])
    P --> F
    F --> CL
    CL --> Q1
    Q1 -->|"yes"| M
    Q1 -->|"no"| Q2
    Q2 -->|"yes"| H
    Q2 -->|"no"| Q3
    Q3 -->|"yes"| I
    Q3 -->|"no"| Q4
    Q4 -->|"yes"| BC
    Q4 -->|"no"| EX
    M --> MV
    H --> MV
    I --> MV
    BC --> MV
    EX --> MV
    classDef core fill:#eceef4,stroke:#8a90a6,color:#222,stroke-width:1.5px
    classDef mA fill:#cfe3ff,stroke:#4a86d6,color:#222,stroke-width:1.5px
    classDef mC fill:#d3f2d0,stroke:#4fa84a,color:#222,stroke-width:1.5px
    classDef term fill:#dddddd,stroke:#777777,color:#222,stroke-width:1.5px
    classDef test fill:#f2f2f7,stroke:#8a90a6,color:#222,stroke-width:1.5px
    class F core
    class M,H,I,BC,EX mA
    class CL mC
    class P,MV term
    class Q1,Q2,Q3,Q4 test
```

*Figure 5. Decision cycle of an `IntelligentAgent` on every turn.* (image version: `diagrams/05_agent_cycle.png`)

---

## 4. Experiments

The experiments share one pipeline, from training data to charts and replays:

```mermaid
flowchart LR
    LG["Synthetic logs<br/>(policy + noise)"]
    TR["ID3 training<br/>4 personalities"]
    AG["Agents<br/>tree + A* + Minimax"]
    EXPS["Experiments<br/>1 personalities<br/>2 spending vs score<br/>3 escape x buying"]
    GM["Seeded games<br/>(200 per experiment)"]
    RC["Recorder<br/>per-turn data"]
    CSV["CSV results"]
    PNG["Charts (PNG)"]
    JS["Recording (JSON)"]
    GU["GUI replay"]
    LG --> TR
    TR --> AG
    AG --> GM
    EXPS -->|"rosters"| GM
    GM --> RC
    RC --> CSV
    RC --> PNG
    RC --> JS
    JS --> GU
    classDef core fill:#eceef4,stroke:#8a90a6,color:#222,stroke-width:1.5px
    classDef data fill:#fff6c8,stroke:#c9a800,color:#222,stroke-width:1.5px
    classDef mC fill:#d3f2d0,stroke:#4fa84a,color:#222,stroke-width:1.5px
    classDef ui fill:#e6e0fa,stroke:#7b68c8,color:#222,stroke-width:1.5px
    class AG,GM,RC core
    class LG,CSV,PNG,JS data
    class TR mC
    class EXPS,GU ui
```

*Figure 6. Experiment pipeline.* (image version: `diagrams/06_experiment_pipeline.png`)

### 4.1 Personalities

Four trees are trained on logs generated by different economic policies, and each personality also has its own bidding multipliers and its own definition of "rich":

| Personality | `mult_a` | `mult_b` | "Rich" from | Style |
|---|---|---|---|---|
| Agressiu | 1.6 | 0.9 | 20 coins | Buys survival tools readily |
| Equilibrat | 1.4 | 0.6 | 30 coins | Balanced baseline |
| Conservador | 1.1 | 0.4 | 45 coins | Buys only when really needed |
| Estalviador | 1.0 | 0.4 | 60 coins | Hoards coins, almost never buys |

### 4.2 Metrics and charts

`experiments.py` records games and produces charts and CSV files for three studies. Results below come from 200 games each, on the default configuration:

| Study | Finding |
|---|---|
| **Which personality survives best?** | Agressiu: 118 points and 56% survival. The other three reach between 111 and 114 points, within about two standard errors of each other. |
| **Does spending more at auction pay off?** | Barely. The Pearson correlation between coins spent and final score is **0.10**. |
| **Does good escape make buying unnecessary?** (2×2 design: Minimax or greedy flight × buys or never buys) | **Minimax without buying wins clearly:** 135 points, richest wallet, wins 60% of games. Minimax with buying scores 114, greedy without buying 105 and greedy with buying 99. |

In this setup, a strong escape algorithm is worth more than purchased advantages.

### 4.3 Reproducibility

Games are fully deterministic for a given seed (all randomness uses seeded generators), so every experiment can be repeated exactly.

---

## 5. Graphical interface

`gui.py` is a Tkinter application with a side panel that shows, in real time:

- the maze, ghosts, coins, pending items and each Pacman's shield and power-up status;
- one card per agent with lives, wallet, coins spent, auctions won, **the class the decision tree currently assigns (A/B/C)** and **the exact path of feature tests the tree followed to reach it**;
- an auction panel with every agent's bid, the strategy and valuation behind it, the winner (★) and the price paid, plus a history list.

Games can be recorded to JSON and replayed later, with step-forward and step-back controls.

---

## 6. Testing

The `tests/` folder contains **142 unit tests** using only the standard library. They cover:

- search algorithms, including A\* optimality against BFS, danger avoidance, and a 40-turn Minimax survival simulation against a chasing ghost;
- both auction types, including the Vickrey property that the price does not depend on the winner's own bid;
- ID3 (entropy, root-feature selection, depth limit, unseen values, persistence);
- game rules (collisions, lives, shields, power-ups, items, auctions, scoring, determinism and invariants over a full game);
- agent decisions, bidding behaviour, metrics, the recorder and the replay logic.

---

## 7. Running the project

```bash
python main.py --games 100                    # basic tournament
python train_tree.py                          # train the base decision tree
python gui.py                                 # live GUI
python gui.py --record game.json              # record a game
python gui.py --replay game.json              # replay a recorded game
python experiments.py --games 200 --out results   # experiments + charts
python -m unittest discover -s tests -t . -v      # unit tests
```

---

## 8. Limitations and future work

- **Training data is synthetic.** The trees learn the rules of a generated expert policy (plus noise), not an empirically optimal strategy. Training on logs of real games, or on game outcomes, would be the natural next step.
- **The game engine is custom** and deliberately minimal. It can be replaced by a full Pacman engine by adapting the perception (`Game.make_view`) and the agents' `act` method.
- **Results depend on the configuration.** All studies use one maze and one set of parameters. In particular, the Conservador and Estalviador agents almost never buy, because they rarely reach their own "rich" threshold by the first auction. Lowering those thresholds would change the outcome.
- **Power Pellets are rarely decisive.** Almost no ghosts are eaten in practice, so their value in the economy is probably overestimated.
- **Possible extensions:** learning bid multipliers by reinforcement learning, trading between agents, partial observability of ghosts, and larger or procedurally generated mazes.
