use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::cmp::{max, min};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

const ROWS: usize = 9;
const COLS: usize = 9;
const MAX_SCORE: i32 = 1_000_000;
const MIN_SCORE: i32 = -MAX_SCORE;

// Define the possible flags for entries
#[derive(Debug, Clone, Copy)]
enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

// Structure for a transposition table entry
#[derive(Debug, Clone, Copy)]
struct TTEntry {
    best_move: Option<(usize, usize, usize, usize)>,
    eval: i32,
    depth: i32,
    flag: TTFlag,
}

type TranspositionTable = HashMap<u64, TTEntry>;

#[pyfunction]
fn get_best_move(
    _py: Python,
    board: &PyArray2<i8>,
    player: i8,
    depth: i32,
) -> PyResult<(i32, usize, usize, usize, usize)> {
    // Safely access the board data
    let board_readonly = board.readonly();
    let board_state = board_readonly.as_array();

    // Validate board shape
    if board_state.shape() != [ROWS, COLS] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid board shape. Expected 9x9 array.",
        ));
    }

    // Convert the ndarray to Vec<Vec<i8>>
    let board_state: Vec<Vec<i8>> = board_state
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    let valid_moves = get_valid_moves(&board_state, player);
    
    if valid_moves.len() == 1 {
        return Ok((
            404, // Placeholder for evaluation
            valid_moves[0].0,
            valid_moves[0].1,
            valid_moves[0].2,
            valid_moves[0].3,
        ));
    }

    // Initialize the Transposition Table
    let mut tt: TranspositionTable = HashMap::new();

    // Call the Negamax algorithm with the Transposition Table
    let (mut best_score, best_move) = negamax(
        &board_state,
        depth,
        player,
        MIN_SCORE,
        MAX_SCORE,
        &mut tt,
    );

    best_score = -player as i32 * best_score;

    println!("Best move negamax: {:?}", best_move);

    // Return the best move and evaluation score if available
    if let Some((from_row, from_col, to_row, to_col)) = best_move {
        Ok((best_score, from_row, from_col, to_row, to_col))
    } else {
        println!("{}", best_score);
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No valid moves available for the AI.",
        ))
    }
}

#[pyfunction]
fn get_valid_moves_python(
    board: &PyArray2<i8>,
    player: i8,
) -> PyResult<Vec<(usize, usize, usize, usize)>> {
    // Convert the ndarray to Vec<Vec<i8>>
    let board_readonly = board.readonly();
    let board_state: Vec<Vec<i8>> = board_readonly
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    Ok(get_valid_moves(&board_state, player))
}

#[pyfunction]
fn evaluate_board_python(board: &PyArray2<i8>) -> i32 {
    // Convert the ndarray to Vec<Vec<i8>>
    let board_readonly = board.readonly();
    let board_state: Vec<Vec<i8>> = board_readonly
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    evaluate_board(&board_state)
}

fn negamax(
    board: &Vec<Vec<i8>>,
    depth: i32,
    player: i8,
    mut alpha: i32,
    mut beta: i32,
    tt: &mut TranspositionTable,
) -> (i32, Option<(usize, usize, usize, usize)>) {
    // Generate a unique key for the current board
    let key = board_to_key(board);
    let old_alpha = alpha;
    let mut old_best_move: Option<(usize, usize, usize, usize)> = None;

    // Check if the position is already in the transposition table
    if let Some(entry) = tt.get(&key) {
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return (entry.eval, entry.best_move),
                TTFlag::LowerBound => alpha = max(alpha, entry.eval),
                TTFlag::UpperBound => {
                    // To prevent overflow, ensure that `min` is used correctly
                    // Note: Ensure you have `use std::cmp::min;` at the top

                    //Chat GPT made this, I'm not sure about it:
                    // let new_beta = min(beta, entry.score);
                    // if new_beta < beta {
                    //     return (entry.score, None);
                    // }

                    //This is from the class pseudocode:
                    beta = min(beta, entry.eval);
                },
            }
            if alpha >= beta {
                return (entry.eval, None);
            }
        } 
        if entry.best_move.is_some() {
            old_best_move = entry.best_move;
        }
    }

    if depth == 0 || is_game_over(board, player) {
        let eval = -player as i32 * evaluate_board(board);
        return (eval, None);
    }

    let mut max_eval = -std::i32::MAX; //It is negative infinity and not MIN_SCORE just in case int gets compared with an actual MIN_SCORE
    let mut best_move = None;

    let mut moves = get_valid_moves(board, player);

    if let Some(best_move_from_tt) = old_best_move {
        if let Some(pos) = moves.iter().position(|&m| m == best_move_from_tt) {
            moves.swap(0, pos); // Move the best_move to the front
        }
    }

    for m in moves {
        let mut new_board = board.clone();
        let capture = make_move(&mut new_board, player, m);
        let new_depth = if capture { depth } else { depth - 1 }; // Captures don't decrease depth

        let (eval, _) = negamax(&new_board, new_depth, -player, -beta, -alpha, tt);
        let eval = -eval;

        if eval > max_eval {
            max_eval = eval;
            best_move = Some(m);
        }
        alpha = max(alpha, eval);
        if alpha >= beta {
            break; // Beta cutoff
        }
    }

    // Determine the flag for the transposition table entry
    let flag = if max_eval <= old_alpha {
        TTFlag::UpperBound
    } else if max_eval >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };

    // Store the evaluation in the transposition table
    let entry = TTEntry {
        best_move,
        eval: max_eval,
        depth,
        flag,
    };
    tt.insert(key, entry);

    (max_eval, best_move)
}

fn board_hash(board: &Vec<Vec<i8>>) -> u64 {
    let mut hasher = DefaultHasher::new();
    board.hash(&mut hasher);
    hasher.finish()
}

fn board_to_key(board: &Vec<Vec<i8>>) -> u64 {
    board_hash(board)
}

fn get_valid_moves(board: &Vec<Vec<i8>>, player: i8) -> Vec<(usize, usize, usize, usize)> {
    let captures = get_possible_captures(board, player);
    if !captures.is_empty() {
        captures
    } else {
        get_all_possible_moves(board, player)
    }
}

fn get_possible_captures(
    board: &Vec<Vec<i8>>,
    player: i8,
) -> Vec<(usize, usize, usize, usize)> {
    let mut captures = Vec::new();
    let direction: i32 = if player == 1 { 1 } else { -1 };

    for i in 0..ROWS {
        for j in 0..COLS {
            if board[i][j] == player {
                let delta_rows = 2 * direction;
                let enemy_row = i as i32 + delta_rows / 2;
                let enemy_cols = [j as i32 - 1, j as i32 + 1];
                let land_row = i as i32 + delta_rows;
                let land_cols = [j as i32 - 2, j as i32 + 2];

                for k in 0..2 {
                    let enemy_col = enemy_cols[k];
                    let land_col = land_cols[k];
                    if enemy_row >= 0
                        && enemy_row < ROWS as i32
                        && enemy_col >= 0
                        && enemy_col < COLS as i32
                        && land_row >= 0
                        && land_row < ROWS as i32
                        && land_col >= 0
                        && land_col < COLS as i32
                    {
                        if board[enemy_row as usize][enemy_col as usize] == -player
                            && board[land_row as usize][land_col as usize] == 0
                        {
                            captures.push((
                                i,
                                j,
                                land_row as usize,
                                land_col as usize,
                            ));
                        }
                    }
                }
            }
        }
    }

    captures
}

fn get_all_possible_moves(
    board: &Vec<Vec<i8>>,
    player: i8,
) -> Vec<(usize, usize, usize, usize)> {
    let mut moves = Vec::new();
    let direction: i32 = if player == 1 { 1 } else { -1 };

    for i in 0..ROWS {
        for j in 0..COLS {
            if board[i][j] == player {
                // Forward move
                let fwd_row = i as i32 + direction;
                if fwd_row >= 0 && fwd_row < ROWS as i32 && board[fwd_row as usize][j] == 0 {
                    moves.push((i, j, fwd_row as usize, j));
                }
                // Side moves
                for delta_col in [-1, 1].iter() {
                    let side_col = j as i32 + delta_col;
                    if side_col >= 0
                        && side_col < COLS as i32
                        && board[i][side_col as usize] == 0
                    {
                        moves.push((i, j, i, side_col as usize));
                    }
                }
            }
        }
    }

    moves
}

fn make_move( // Returns true if a capture was made
    board: &mut Vec<Vec<i8>>,
    player: i8,
    mv: (usize, usize, usize, usize),
) -> bool {
    let (from_row, from_col, to_row, to_col) = mv;
    board[from_row][from_col] = 0;
    board[to_row][to_col] = player;

    // Check for capture
    if (from_row as i32 - to_row as i32).abs() == 2 {
        let captured_row = (from_row + to_row) / 2;
        let captured_col = (from_col + to_col) / 2;
        board[captured_row][captured_col] = 0;
        return true; // Capture
    }
    return false; // No capture
}

fn evaluate_board(board: &Vec<Vec<i8>>) -> i32 {
    if is_game_over(board, 1) {
        return MAX_SCORE;
    } 
    if is_game_over(board, -1) {
        return MIN_SCORE;
    }
    // Calculate the score based on the maximizer's perspective
    let mut score = 0;
    let mut length_triangle_white: usize = ROWS - 1;
    let mut length_triangle_black: usize = ROWS - 1;
    let mut triangle_found = false;
    for i in 0..ROWS {
        for j in 0..COLS {
            match board[i][j] {
                -1 => {
                    if triangle_to_win(board, -1, i, j) {
                        length_triangle_white = std::cmp::min(length_triangle_white, i);
                        triangle_found = true;
                        continue;
                    }
                    if !triangle_found {
                        score += 10;
                        score += (ROWS - i - 1) as i32;
                        score += (j as i32 - 4).abs();
                    }
                },
                1 => {
                    if triangle_to_win(board, 1, i, j) {
                        length_triangle_black = std::cmp::min(length_triangle_black, ROWS - i - 1 as usize);
                        triangle_found = true;
                        continue;
                    }
                    if !triangle_found {
                        score -= 10;
                        score -= i as i32;
                        score -= (j as i32 - 4).abs();
                    }
                },
                _ => (),
            }
        }
    }
    
    if length_triangle_black < length_triangle_white {
        // println!("Triangle for black");
        return MIN_SCORE;
    } else if length_triangle_black > length_triangle_white {
        // println!("Triangle for white");
        return MAX_SCORE;
    }
    score
}

fn triangle_to_win(board: &Vec<Vec<i8>>, player: i8, i: usize, j: usize) -> bool {
    let i = i as isize;
    let j = j as isize;
    let rows = ROWS as isize;
    let cols = COLS as isize;

    let (i1, i2) = if player == 1 {
        (i + 1, rows - 1)
    } else {
        if i == 0 {
            println!("i is zero: {}", i);
        }
        (0, if i > 0 { i - 1 } else { 0 })
    };

    for n in i1..=i2 {
        let delta = player as isize * (n - i);
        let j1 = (j - delta).max(0).min(cols - 1);
        let j2 = (j + delta).max(0).min(cols - 1);
        if j2 < 0 || j1 < 0 {
            println!("j1 or j2 is less than zero: {}, {}", j1, j2);
        }
        for m in j1..=j2 {
            if board[n as usize][m as usize] == -player {
                return false;
            }
        }
    }
    true
}

fn is_game_over(board: &Vec<Vec<i8>>, player: i8) -> bool {
    // Check if any of player's pieces reached the opposite end
    if is_winner(board, -player) {
        return true;
    }
    // Or no valid moves left
    get_valid_moves(board, player).is_empty()
}

fn is_winner(board: &Vec<Vec<i8>>, player: i8) -> bool {
    let target_row = if player == 1 { ROWS - 1 } else { 0 };
    for j in 0..COLS {
        if board[target_row][j] == player {
            return true;
        }
    }
    false
}

/// A Python module implemented in Rust.
#[pymodule]
fn fianco_brain(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_best_move, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_board_python, m)?)?;
    m.add_function(wrap_pyfunction!(get_valid_moves_python, m)?)?;
    Ok(())
}
