# Fianco

A digital implementation of **Fianco**, developed as part of the *Intelligent Search and Games* course in the Master's in Artificial Intelligence at Maastricht University. The project was graded through a tournament in which each student’s AI competed against the others, where I secured **1st place**. The repository provides a fully playable Python version of the game, powered by a Rust-based AI backend, along with a precompiled `.exe` for Windows users who simply want to play.

---

## Rules of Fianco

- GOAL: A player wins if they place one of their stones on the opponent’s last row.

- Each turn a player must move one of their stones. A stone may:

    - Move forwards or sideways to an adjacent empty cell.

    - Capture by jumping diagonally forward over an enemy stone, landing on the immediate empty cell. Capturing is mandatory, but only one capture is allowed per turn (no multi-captures).

- In case a player has lost all its pieces, this player loses the game.

- In case a player cannot make a move (aka stalemate), the player loses the game.

- Threefold repetition constitutes a draw.

---

## How to Play

- The game starts with a **setup screen** where you select who controls each side:

  - **Human** – controlled by you.

  - **AI** – controlled by the computer.

    - If AI is selected, you can choose between:

      - **Depth mode** – the AI searches a fixed number of moves ahead. 

      - **Time mode** – the AI searches as deeply as possible within the set number of seconds.

- Once a new game is started, the 'Human' player(s) are controlled by the user.

- When a piece is selected, valid moves are highlighted.

---

## Implementation Details

The AI for **Fianco** is based on the **Negamax algorithm**, enhanced with several techniques to improve performance and decision-making:

- **Transposition Table with Zobrist Hashing**: Efficiently avoids recalculating previously explored positions by storing and retrieving board states using unique hash values.

- **Iterative Deepening**: Gradually increases the search depth, ensuring that the AI can return the best result found so far even under strict time constraints.

- **Quiescence Search**: Extends the search selectively during "noisy" positions. For this project, a move that captures a piece does not count for depth calculation, as these moves are forced.

- **Evaluation Function**:

  - Rewards pieces that move toward the sides of the board, reinforcing strong positional play.  

  - Detects **triangular structures** around each piece, identifying "passed" pieces even if they lie beyond the current search depth.

---

## Installation / Running

### Option 1: Use the `.exe` (Recommended)
> Windows only
1. Download `dist/Fianco.exe` from the repo.  
2. Run the .exe file. No Python installation required.

### Option 2: Running From Source (Without the `.exe`)

This project has a **Python frontend** and a **Rust-powered AI backend**. To run from source on a new machine you must:
1) set up Python,  
2) install the Rust toolchain, and  
3) build the native Rust extension/library so Python can use it.

#### 1) Prerequisites

- **Python**: 3.11+
- **Rust toolchain**: 1.81 install via [rustup](https://rustup.rs/)
  - Windows users: install the **MSVC** toolchain and **Build Tools for Visual Studio** (C++ workload).
- **Git**: to clone the repository


#### 2) Clone and create a virtual environment

```bash
git clone https://github.com/MarioRealEn/Fianco.git
cd Fianco
```

Create & activate a virtual environment (choose one):

- On Windows (CMD):
```bash
python -m venv .venv
.venv\Scripts\activate
```

- On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3) Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


---

#### 4) Build the Rust backend (PyO3 + Maturin)

```bash
python -m pip install maturin # should be installed already form the requirements
cd brain          # adjust path if your Rust crate is elsewhere
maturin develop --release
cd ..
```

---

#### 5) Run the game

```bash
python main.py
```

---