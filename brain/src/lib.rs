use numpy::PyArray2;
use pyo3::prelude::*;
use std::cmp::{max, min};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{Instant, Duration};
use std::sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}};
use std::thread;
use ndarray::ArrayView2;

const ROWS: usize = 9;
const COLS: usize = 9;
const MAX_SCORE: i32 = 1_000_000;
const MIN_SCORE: i32 = -MAX_SCORE;
const DRAW_SCORE: i32 = -30;
const WIN_BY_TRIANGLE: i32 = 50_000;
const MAX_TT_SIZE: usize = 30_000_000; // Adjust as needed

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
type Board = [[i8; COLS]; ROWS];

#[pyclass]
struct FiancoAI {
    tt: Arc<RwLock<TranspositionTable>>,
    zobrist_table: Arc<Vec<Vec<[u64; 2]>>>, // [ROWS][COLS][2]
    current_hash_key: u64,
    hash_history: Vec<u64>,
    ai_player: i8,
    root_move_scores: HashMap<(usize, usize, usize, usize), i32>,
    pondering_thread: Option<thread::JoinHandle<()>>,
    pondering_stop_flag: Arc<AtomicBool>,
}

#[pymethods]
impl FiancoAI {
    #[new]
    fn new(ai_player: i8) -> Self {
        // Initialize the zobrist_table with random numbers
        let mut rng = StdRng::seed_from_u64(0);
        let mut zobrist_table: Vec<Vec<[u64; 2]>> = vec![vec![[0u64; 2]; COLS]; ROWS];
        for i in 0..ROWS {
            for j in 0..COLS {
                for k in 0..2 {
                    zobrist_table[i][j][k] = rng.gen::<u64>();
                }
            }
        }

        FiancoAI {
            tt: Arc::new(RwLock::new(HashMap::new())),
            zobrist_table: Arc::new(zobrist_table),
            current_hash_key: 0,
            hash_history: Vec::new(),
            ai_player: ai_player,
            root_move_scores: HashMap::new(),
            pondering_thread: None,
            pondering_stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    fn get_best_move(
        &mut self,
        _py: Python,
        board: &PyArray2<i8>,
        player: i8,
        max_depth: i32,
        max_time: u64,
        pondering: bool,
    ) -> PyResult<(i32, Vec<(usize, usize, usize, usize)>)> {
        // Stop pondering if it's running
        self.stop_pondering().ok();

        // println!("Transposition Table size: {}", self.tt.write().unwrap().len());

        // Safely access the board data
        let mut best_score = 505;
        let mut pv = Vec::new();
        let mut depth_results: Vec<(i32, i32, Vec<(usize, usize, usize, usize)>)> = Vec::new();

        let start_time = Instant::now();
        let max_time = Duration::new(max_time, 0);

        // Convert the ndarray to Board
        let mut board_state: Board = pyarray_to_board(board)?;

        // Get valid moves
        let valid_moves = get_valid_moves(&board_state, player);

        if valid_moves.len() == 1 {
            return Ok((
                404, // Placeholder for evaluation
                vec![valid_moves[0]],
            ));
        }

        if valid_moves.is_empty() {
            println!("I checked and there are no valid moves")
        }

        self.current_hash_key = self.compute_hash_key(&board_state);

        // Push the current hash key onto the stack
        self.hash_history.push(self.current_hash_key);

        // Initialize root_move_scores
        self.root_move_scores.clear();

        // Check TT size and clear if necessary
        {
            let tt_size = {
                let tt = self.tt.read().unwrap();
                tt.len()
            }; // Read lock is released here
        
            // If the table size exceeds the maximum, acquire a write lock to clear it
            if tt_size >= MAX_TT_SIZE {
                let mut tt = self.tt.write().unwrap();
                tt.clear();
                println!("Transposition Table cleared.");
                // Write lock is released when `tt` goes out of scope
            }
        }

        for depth in 1..=max_depth {
            if start_time.elapsed() >= max_time {
                println!("Time limit reached. Breaking out of the search loop.");
                break;
            }

            // Copy the current hash key
            let mut hash_key = self.current_hash_key;

            // Call the Negamax algorithm with the Transposition Table
            let result = self.negamax(
                &mut board_state,
                depth,
                player,
                MIN_SCORE,
                MAX_SCORE,
                &mut hash_key,
                true,
                &start_time,
                max_time,
                false,
            );

            match result {
                Ok((score, pv_current)) => {
                    best_score = -player as i32 * score;
                    pv = pv_current.clone();
                    depth_results.push((depth, best_score, pv_current.clone()));
                    println!("Depth {}: Best Score = {}, PV = {:?}", depth, best_score, pv);
                }
                Err(_) => {
                    // Time limit reached during negamax; break out of the loop
                    println!("Time limit reached during negamax. Breaking out of the search loop.");
                    break;
                }
            }
        }

        println!("Time: {}", start_time.elapsed().as_secs_f64());

        let loss_in_sight = depth_results.iter().any(|&(_, score, _)| player as i32 * score >= WIN_BY_TRIANGLE);
        let max_score_achieved = depth_results.iter().any(|&(_, score, _)| score == -player as i32 * MAX_SCORE);

        // Find the highest depth where score != MIN_SCORE
        let mut best_pv = None;
        let mut pv_last_iter = Vec::new();
        for (_, score, pv_candidate) in depth_results.iter().rev() {
            if loss_in_sight {
                if player as i32 * *score < WIN_BY_TRIANGLE {
                    best_pv = Some(pv_candidate.clone());
                    break;
                }
            }
            if max_score_achieved && !loss_in_sight {
                best_pv = Some(pv_candidate.clone());
                if *score != -player as i32 * MAX_SCORE {
                    if !pv_last_iter.is_empty() {
                        best_pv = Some(pv_last_iter.clone());
                    }
                    break;
                }
                pv_last_iter = pv_candidate.clone();
            }
        }
        if let Some(best_pv) = best_pv {
            // Update pv to avoid the move leading to a forced loss
            pv = best_pv;
        } else {
            // All depths resulted in MIN_SCORE or MAX_SCORE; pv remains as is
        }
        // Return the best move and evaluation score if available
        if let Some(&(_from_row, _from_col, _to_row, _to_col)) = pv.first() {
            // println!("Transposition Table size: {}", self.tt.lock().unwrap().len());
            if pondering {
                // Simulate the move to get the updated board state
                let mut new_board_state = board_state.clone();
                let mut new_hash_key = self.current_hash_key;
                self.make_move(&mut new_board_state, player, pv[0], &mut new_hash_key);
    
                // Start pondering for the next move
                self.start_pondering(
                    &new_board_state,
                    -player,
                    max_depth,
                    Duration::new(20, 0),
                )?;
            }
            Ok((best_score, pv))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No valid moves available for the AI.",
            ))
        }
    }

    // #[pyfunction]
    // fn get_valid_moves_python(
    //     board: &PyArray2<i8>,
    //     player: i8,
    // ) -> PyResult<Vec<(usize, usize, usize, usize)>> {
    //     // Convert the ndarray to Vec<Vec<i8>>
    //     let board_readonly = board.readonly();
    //     let board_state: Vec<Vec<i8>> = board_readonly
    //         .as_array()
    //         .outer_iter()
    //         .map(|row| row.to_vec())
    //         .collect();

    //     Ok(get_valid_moves(&board_state, player))
    // }

    fn stop_pondering(&mut self) -> PyResult<()> {
        // Signal the pondering thread to stop
        if let Some(_) = self.pondering_thread {
            self.pondering_stop_flag.store(true, Ordering::Relaxed);
            // Wait for the thread to finish
            if let Some(handle) = self.pondering_thread.take() {
                let _ = handle.join();
            }
        }
        Ok(())
    }

    // fn evaluate_board_python(&mut self, board: &PyArray2<i8>, player: i8) -> i32 {
    //     // Convert the ndarray to Vec<Vec<i8>>
    //     let mut board_state: Board = pyarray_to_board(board)?;

    //     evaluate_board(&board_state, player)
    // }

    #[pyo3(name = "get_tt_size")]
    fn get_tt_size(&self) -> PyResult<usize> {
        let tt = self.tt.read().unwrap();
        let size = tt.len();
        drop(tt);
        Ok(size)
        
    }
}

impl FiancoAI {
    fn negamax(
        &mut self,
        board: &mut Board,
        depth: i32,
        player: i8,
        mut alpha: i32,
        mut beta: i32,
        hash_key: &mut u64,
        is_root: bool,
        start_time: &Instant,
        max_time: Duration,
        pondering: bool, //If this is a pondering run, set to True
    ) -> Result<(i32, Vec<(usize, usize, usize, usize)>), ()> {
        // Check the stop flag
        if pondering && self.pondering_stop_flag.load(Ordering::Relaxed) {
            println!("Stop signal received during recursive call.");
            return Err(());
        }

        let key = *hash_key;
        let old_alpha = alpha;
        let mut old_best_move: Option<(usize, usize, usize, usize)> = None;

        if start_time.elapsed() >= max_time {
            println!("Time limit reached during recursive call.");
            return Err(());
        }

        // Count how many times the current position has occurred in the current path
        let repetitions = self.hash_history.iter().filter(|&&k| k == key).count();

        // Check for threefold repetition
        if repetitions >= 3 {
            return Ok((-self.ai_player as i32 * DRAW_SCORE, Vec::new())); // Return a score indicating a draw
        } else if repetitions == 0 {
            // Transposition Table lookup
            let tt = self.tt.read().unwrap();
            if let Some(entry) = tt.get(&key) {
                if entry.depth >= depth {
                    match entry.flag {
                        TTFlag::Exact => {
                            let mut pv = Vec::new();
                            if let Some(best_move) = entry.best_move {
                                pv.push(best_move);
                            }
                            return Ok((entry.eval, pv));
                        }
                        TTFlag::LowerBound => alpha = max(alpha, entry.eval),
                        TTFlag::UpperBound => beta = min(beta, entry.eval),
                    }
                    if alpha >= beta {
                        return Ok((entry.eval, Vec::new()));
                    }
                }
                if entry.best_move.is_some() {
                    old_best_move = entry.best_move;
                }
            }
            drop(tt); // Release the lock
        }

        // Check for depth or game over
        if depth == 0 || is_game_over(board, player) {
            let eval = -player as i32 * evaluate_board(board, player);
            return Ok((eval, Vec::new()));
        }

        let mut max_eval = -std::i32::MAX;
        let mut best_pv = Vec::new();

        // Get valid moves
        let mut moves = get_valid_moves(board, player);

        if is_root {
            // Sort moves based on root_move_scores
            moves.sort_by_cached_key(|&m| {
                // Use negative scores to sort in descending order
                player as i32 * (self.root_move_scores.get(&m).cloned().unwrap_or(0))
            });
        } else {
            // At non-root nodes, optionally use TT best move
            if let Some(best_move_from_tt) = old_best_move {
                if let Some(pos) = moves.iter().position(|&m| m == best_move_from_tt) {
                    moves.swap(0, pos); // Move the best_move to the front
                }
            }
        }

        // Iterate over the moves
        for m in moves {
            // Make the move and update hash key
            let capture = self.make_move(board, player, m, hash_key);

            let new_depth = if capture { depth } else { depth - 1 };

            // Recursive call
            let result = self.negamax(
                board,
                new_depth,
                -player,
                -beta,
                -alpha,
                hash_key,
                false,
                start_time,
                max_time,
                pondering,
            );

            // Undo the move and restore hash key
            self.undo_move(board, player, m, capture, hash_key);

            match result {
                Ok((eval, pv)) => {
                    let eval = -eval;

                    if eval > max_eval {
                        max_eval = eval;
                        best_pv = pv;
                        best_pv.insert(0, m); // Prepend the current move to the PV
                    }
                    alpha = max(alpha, eval);
                    if alpha >= beta {
                        break; // Beta cutoff
                    }
                }
                Err(_) => {
                    // Time limit reached or stop signal received during recursive call
                    return Err(());
                }
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
            best_move: if best_pv.is_empty() { None } else { Some(best_pv[0]) },
            eval: max_eval,
            depth,
            flag,
        };

        let mut tt = self.tt.write().unwrap();
        tt.insert(key, entry);
        drop(tt); // Release the lock

        if is_root && !best_pv.is_empty() {
            // At root, store the move's score for ordering
            self.root_move_scores.insert(best_pv[0], max_eval);
        }

        Ok((max_eval, best_pv))
    }

    fn start_pondering(
        &mut self,
        board: &Board,
        player: i8,
        max_depth: i32,
        max_duration: Duration,
    ) -> PyResult<()> {
        // Check if pondering is already running
        if self.pondering_thread.is_some() {
            return Ok(()); // Already pondering
        }
        println!("Starting pondering thread...");

        // Reset the stop flag
        self.pondering_stop_flag.store(false, Ordering::Relaxed);

        // Clone necessary data for the thread
        let mut board_state = board.clone();
        let tt_clone = Arc::clone(&self.tt);
        let zobrist_table_clone = Arc::clone(&self.zobrist_table);
        let stop_flag = Arc::clone(&self.pondering_stop_flag);
        let ai_player = self.ai_player;

        // Start the pondering thread
        self.pondering_thread = Some(thread::spawn(move || {
            let mut ai = FiancoAI {
                tt: tt_clone,
                zobrist_table: zobrist_table_clone,
                current_hash_key: 0,
                hash_history: Vec::new(),
                ai_player,
                root_move_scores: HashMap::new(),
                pondering_thread: None,
                pondering_stop_flag: stop_flag,
            };

            let start_time = Instant::now();

            ai.current_hash_key = ai.compute_hash_key(&board_state);

            // Run the search loop
            for depth in 1..=max_depth {
                if ai.pondering_stop_flag.load(Ordering::Relaxed) {
                    // Stop signal received
                    break;
                }
                if start_time.elapsed() >= max_duration {
                    break;
                }

                // Copy the current hash key
                let mut hash_key = ai.current_hash_key;

                // Call negamax
                let result = ai.negamax(
                    &mut board_state,
                    depth,
                    player,
                    MIN_SCORE,
                    MAX_SCORE,
                    &mut hash_key,
                    true,
                    &start_time,
                    max_duration,
                    true
                );

                println!("Pondering depth {}: {:?}", depth, result);
                // In the pondering thread, we may not need to store the result
            }
        }));

        Ok(())
    }


    fn compute_hash_key(&self, board: &Board) -> u64 {
        let mut hash_key = 0u64;
        for i in 0..ROWS {
            for j in 0..COLS {
                let piece = board[i][j];
                if piece != 0 {
                    let piece_index = if piece == -1 { 0 } else { 1 };
                    hash_key ^= self.zobrist_table[i][j][piece_index];
                }
            }
        }
        hash_key
    }
    fn make_move(
        &mut self,
        board: &mut Board,
        player: i8,
        mv: (usize, usize, usize, usize),
        hash_key: &mut u64,
    ) -> bool {
        let (from_row, from_col, to_row, to_col) = mv;

        let piece_index = if player == -1 { 0 } else { 1 };

        // XOR out the piece from its original position
        *hash_key ^= self.zobrist_table[from_row][from_col][piece_index];

        // Remove the piece from original position
        board[from_row][from_col] = 0;

        // XOR in the piece at the new position
        *hash_key ^= self.zobrist_table[to_row][to_col][piece_index];
        
        // Push the current hash key onto the stack
        self.hash_history.push(*hash_key);
        // println!("hash_history after make_move: {:?}", self.hash_history);

        // Place the piece at new position
        board[to_row][to_col] = player;

        let mut captured = false;

        // Check for capture
        if (from_row as i32 - to_row as i32).abs() == 2 {
            let captured_row = (from_row + to_row) / 2;
            let captured_col = (from_col + to_col) / 2;
            let captured_piece_index = if -player == -1 { 0 } else { 1 };

            // XOR out the captured piece
            *hash_key ^= self.zobrist_table[captured_row][captured_col][captured_piece_index];

            // Remove the captured piece
            board[captured_row][captured_col] = 0;

            captured = true;
        }

        captured
    }
    fn undo_move(
        &mut self,
        board: &mut Board,
        player: i8,
        mv: (usize, usize, usize, usize),
        captured: bool,
        hash_key: &mut u64,
    ) {
        let (from_row, from_col, to_row, to_col) = mv;

        let piece_index = if player == -1 { 0 } else { 1 };

        self.hash_history.pop(); // Remove the hash key before undoing the move
        // println!("hash_history after undo_move: {:?}", self.hash_history);

        // XOR out the piece from the destination position
        *hash_key ^= self.zobrist_table[to_row][to_col][piece_index];

        // Remove the piece from the destination position
        board[to_row][to_col] = 0;

        // XOR in the piece at the original position
        *hash_key ^= self.zobrist_table[from_row][from_col][piece_index];

        // Place the piece back at the original position
        board[from_row][from_col] = player;

        if captured {
            let captured_row = (from_row + to_row) / 2;
            let captured_col = (from_col + to_col) / 2;
            let captured_piece_index = if -player == -1 { 0 } else { 1 };

            // XOR in the captured piece
            *hash_key ^= self.zobrist_table[captured_row][captured_col][captured_piece_index];

            // Restore the captured piece
            board[captured_row][captured_col] = -player;
        }
    }
}

// fn board_to_key(board: &[Vec<i8>]) -> u64 {
//     let mut hasher = DefaultHasher::new();
//     board.hash(&mut hasher);
//     hasher.finish()
// }

fn pyarray_to_board(py_array: &PyArray2<i8>) -> PyResult<Board> {
    // Obtain an immutable view of the NumPy array
    let binding = py_array.readonly();
    let board_view: ArrayView2<i8> = binding.as_array();

    // Check if the array has the correct shape
    if board_view.shape() != [ROWS, COLS] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid board shape: expected [{} x {}], got {:?}", ROWS, COLS, board_view.shape()),
        ));
    }

    // Initialize the Board
    let mut board: Board = [[0; COLS]; ROWS];

    // Copy data from the NumPy array to the Board
    for i in 0..ROWS {
        for j in 0..COLS {
            board[i][j] = board_view[[i, j]];
        }
    }

    Ok(board)
}

fn get_valid_moves(board: &Board, player: i8) -> Vec<(usize, usize, usize, usize)> {
    let captures = get_possible_captures(board, player);
    if !captures.is_empty() {
        captures
    } else {
        get_all_possible_moves(board, player)
    }
}

fn get_possible_captures(
    board: &Board,
    player: i8,
) -> Vec<(usize, usize, usize, usize)> {
    let mut captures = Vec::new();
    let direction: i32 = if player == 1 { 1 } else { -1 };

    for i in 0..ROWS {
        for j in 0..COLS {
            // let i = ROWS - (m+1);
            // let j = COLS - (n+1);
            if board[i][j] == player {  //CHANGE THIS
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
    board: &Board,
    player: i8,
) -> Vec<(usize, usize, usize, usize)> {
    let mut moves = Vec::new();
    let direction: i32 = if player == 1 { 1 } else { -1 };

    for i in 0..ROWS {
        for j in 0..COLS {
            // let i = ROWS - (m+1);
            // let j = COLS - (n+1);
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

#[inline]
fn evaluate_board(board: &Board, player_to_move: i8) -> i32 {
    // if is_game_over(board, 1) {
    //     return MAX_SCORE;
    // } 
    if is_game_over(board, player_to_move) {
        return player_to_move as i32 * MAX_SCORE;
    }
    // Calculate the score based on the maximizer's perspective
    let mut score = 0;
    let mut length_triangle_white: usize = ROWS;
    let mut length_triangle_black: usize = ROWS;
    let mut triangle_found = false;
    for i in 0..ROWS {
        for j in 0..COLS {
            match board[i][j] {
                -1 => {
                    if triangle_to_win(board, player_to_move,-1, i, j) {
                        length_triangle_white = std::cmp::min(length_triangle_white, i);
                        triangle_found = true;
                        continue;
                    }
                    if !triangle_found {
                        score += 20;
                        score += 3 * (ROWS - i - 1) as i32;
                        score += 2 * (j as i32 - 4).abs();
                    }
                },
                1 => {
                    if triangle_to_win(board, player_to_move, 1, i, j) {
                        length_triangle_black = std::cmp::min(length_triangle_black, ROWS - i - 1 as usize);
                        triangle_found = true;
                        continue;
                    }
                    if !triangle_found {
                        score -= 20;
                        score -= 3 * i as i32;
                        score -= 2 * (j as i32 - 4).abs();
                    }
                },
                _ => (),
            }
        }
    }
    // println!("Triangle for white: {}", length_triangle_white);
    // println!("Triangle for black: {}", length_triangle_black);
    if length_triangle_black != ROWS || length_triangle_white != ROWS {
        return (2 * (length_triangle_black - length_triangle_white) as i32 - player_to_move as i32) * WIN_BY_TRIANGLE;
    }
    score
}

// fn triangle_to_win(board: &[Vec<i8>], player: i8, i: usize, j: usize) -> bool {
//     let i = i as isize;
//     let j = j as isize;
//     let rows = ROWS as isize;
//     let cols = COLS as isize;

//     let (i1, i2) = if player == 1 {
//         (i + 1, rows - 1)
//     } else {
//         // if i == 0 {
//         //     println!("i is zero: {}", i);
//         // }
//         (0, if i > 0 { i - 1 } else { 0 })
//     };

//     for n in i1..=i2 {
//         let delta = player as isize * (n - i ) - 1;
//         let j1 = (j - delta).max(0).min(cols - 1);
//         let j2 = (j + delta).max(0).min(cols - 1);
//         // if j2 < 0 || j1 < 0 {
//         //     println!("j1 or j2 is less than zero: {}, {}", j1, j2);
//         // }
//         for m in j1..=j2 {
//             if board[n as usize][m as usize] == -player {
//                 return false;
//             }
//         }
//     }
//     true
// }

#[inline]
fn triangle_to_win(board: &Board, player_to_move: i8, player: i8, i: usize, j: usize) -> bool {
    // Convert indices to i32 for calculations
    let i = i as i32;
    let j = j as i32;
    let rows = ROWS as i32;
    let cols = COLS as i32;
    let player_i32 = player as i32; // Cache player as i32

    // Determine the step direction based on the player
    let step_n = -player_i32; // 1 for player 1, -1 for player -1

    // Determine starting and ending row indices based on player
    let (end_n, start_n) = if player == 1 {
        (i + 1, rows - 1)
    } else {
        (i - 1, 0)
    };

    let mut n = start_n;
    let delta_offset = if player_to_move == player {1} else {0}; // Used to adjust delta after the first iteration

    loop {
        let delta = player_i32 * (n - i) - delta_offset;
        let j1 = (j - delta).max(0).min(cols - 1);
        let j2 = (j + delta).max(0).min(cols - 1);

        // Check the triangle row for opponent's pieces
        for m in j1..=j2 {
            if board[n as usize][m as usize] == -player {
                return false;
            }
        }

        // Break the loop if we've reached the end
        if n == end_n {
            break;
        }

        // Move to the next row in the triangle
        n += step_n;
    }
    true
}

fn is_game_over(board: &Board, player: i8) -> bool {
    // Check if any of player's pieces reached the opposite end
    if is_winner(board, -player) {
        return true;
    }
    // Or no valid moves left
    get_valid_moves(board, player).is_empty()
}

fn is_winner(board: &Board, player: i8) -> bool {
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
    m.add_class::<FiancoAI>()?;
    Ok(())
}
