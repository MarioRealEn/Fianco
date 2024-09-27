use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::cmp::max;
use std::usize::MIN;

const ROWS: usize = 9;
const COLS: usize = 9;
const MAX_SCORE: i32 = 1_000_000;
const MIN_SCORE: i32 = -MAX_SCORE;

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
    
    if valid_moves.len()==1{
        return Ok((404, valid_moves[0].0, valid_moves[0].1, valid_moves[0].2, valid_moves[0].3));//idk what to do with the eval
    }

    // Call the Negamax algorithm
    let (mut best_score, best_move) = negamax(
        &board_state,
        depth,
        player,
        MIN_SCORE,
        MAX_SCORE,
    );

    best_score = -player as i32 * best_score;

    println!("Best move python: {:?}", best_move);

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
    beta: i32,
) -> (i32, Option<(usize, usize, usize, usize)>) {
    if depth == 0 || is_game_over(board, player) {
        let eval = -player as i32 * evaluate_board(board);
        return (eval, None);
    }

    let mut max_eval = -std::i32::MAX;
    let mut best_move = None;

    let moves = get_valid_moves(board, player);

    for m in moves {
        let mut new_board = board.clone();
        let capture = make_move(&mut new_board, player, m);
        let new_depth = if capture { depth } else { depth - 1 }; //Captures dont decrease depth

        let (eval, _) = negamax(&new_board, new_depth, -player, -beta, -alpha);
        let eval = -eval;

        if eval > max_eval {
            max_eval = eval;
            best_move = Some(m);
        }
        alpha = max(alpha, eval);
        if eval >= beta {
            break; // Beta cutoff
        }
    }
    // println!("Best move negamax: {:?}", best_move);
    (max_eval, best_move)
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
    let mut length_triangle_white: usize = ROWS-1;
    let mut length_triangle_black: usize = ROWS-1;
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
                    if !triangle_found{
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
                    if !triangle_found{
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
    let i1: isize;
    let i2: isize;
    let mut j1: isize;
    let mut j2: isize;
    let i = i as isize;
    let j = j as isize;
    let rows = ROWS as isize;
    let cols = COLS as isize;

    if player == 1 {
        i1 = i + 1;
        i2 = rows-1;
    } else {
        i1 = 0;
        if i == 0 {
            println!("I ES CERO!! {}", i);
        }
        i2 = i - 1;

    }

    for n in i1..=i2 {
        let delta = player as isize * (n - i);
        j1 = std::cmp::max(0, j - delta);
        j2 = std::cmp::min(cols-1, j + delta);
        if j2 < 0 || j1 < 0 {
            println!("J1 o J2 ES MENOR QUE CERO!! {}, {}", j1, j2);
        }
        for m in j1..=j2 {
            if board[n as usize][m as usize] == -player {
                return false;
            }
        }
    }
    return true;
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

