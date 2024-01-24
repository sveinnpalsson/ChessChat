import chess
import chess.engine
import chess.pgn
import io
import numpy as np
import requests

def fetch_game_moves(url):
    if 'lichess.org' in url:
        game_id = url.split('lichess.org/')[-1].split('/')[0]
    elif 'chess.com' in url:
        raise NotImplementedError('Not yet implemented') #FIXME: handle chess.com games

    api_url = f"https://lichess.org/game/export/{game_id}"
    response = requests.get(api_url)
    if response.status_code == 404:
        raise ValueError("URL not found: ", api_url)
    game_pgn = response.text

    pgn = io.StringIO(game_pgn)
    game = chess.pgn.read_game(pgn)
    
    # Extracting moves
    moves = [move.uci() for move in game.mainline_moves()]
    return moves


def analyze_position(board, engine_name, engine_path, top_n=3):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        num_threads = 24
        engine.configure({"Threads": num_threads})
        if engine_name == 'lc0':
            infos = engine.analyse(board, chess.engine.Limit(time_limit=0.25), multipv=top_n)
        else:
            infos = engine.analyse(board, chess.engine.Limit(depth=12), multipv=top_n)
    return infos

def analyze_game(moves, engine_name, engine_path, time_limit=0.25, depth_limit=12, top_n=5):
    board = chess.Board()
    game_analysis = []
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            for move in moves:
                move = board.parse_san(move)
                # Get evaluations for the top moves and the played move
                if engine_name == 'lc0':
                    infos = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=top_n)
                else:
                    infos = engine.analyse(board, chess.engine.Limit(depth=depth_limit), multipv=top_n)
                
                evaluations = {}
                for info in infos:
                    score = info['score'].relative
                    # Check if the score is a mate score
                    if score.is_mate():
                        # Adjust the mate score representation
                        if score.mate() > 0:  # Checkmate for the current side
                            evaluations[info.get('pv')[0].uci()] = 1000000 - score.mate()
                        else:  # Getting checkmated
                            evaluations[info.get('pv')[0].uci()] = -1000000 - score.mate()
                    else:
                        evaluations[info.get('pv')[0].uci()] = score.score()

                # Sort moves by evaluation
                sorted_moves = sorted(evaluations.items(), key=lambda x: x[1], reverse=True)

                # Make the move on the board
                board.push(move)
                
                # Extract the evaluation for the played move
                played_move_eval = evaluations.get(move)
                if played_move_eval is None and board.is_stalemate():
                    evaluations[move] = 0.0
                    played_move_eval = 0.0
                elif played_move_eval is None and board.is_checkmate():
                    evaluations[move] = 1000000
                    played_move_eval = 1000000
                elif played_move_eval is None:
                    # If the played move is not in the evaluations, analyze it separately
                    if engine_name == 'lc0':
                        played_info = engine.analyse(board, chess.engine.Limit(time=time_limit))
                    else:
                        played_info = engine.analyse(board, chess.engine.Limit(depth=depth_limit))

                    score = played_info['score'].relative
                    if score.is_mate():
                        if score.mate() > 0:
                            evaluations[move] = -1000000 + score.mate()
                        else:
                            evaluations[move] = 1000000 - score.mate()
                    else:
                        evaluations[move] = -score.score()
                    played_move_eval = evaluations.get(move)

                # Extract the top_n moves from sorted_moves
                top_moves = dict(sorted_moves[:top_n])
                # Construct the analysis for this move
                game_analysis.append({
                    'move': move.uci(),
                    'evaluation': played_move_eval,
                    'top_moves': top_moves,
                    'engine_line': [i.uci() for i in infos[0]['pv']]
                })
    except Exception as e:
        return e
    return game_analysis

def board_to_string_with_squares(board):
    board_str = ''
    square_names = [chess.square_name(sq) for sq in chess.SQUARES]
    for rank in range(8, 0, -1):
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)
            piece_char = piece.symbol() if piece else '.'
            board_str += f"{square_names[square]}:{piece_char} "
        board_str += '\n'
    return board_str

def convert_uci_to_san(uci_moves):
    board = chess.Board()
    san_moves = []
    for move in uci_moves:
        move_san = board.san(board.parse_uci(move))
        board.push(board.parse_uci(move))
        san_moves.append(move_san)
    return san_moves    

def board_summary(board, last_move, move_num, current_eval, material_balance_current, move_quality, moves):
    moves = convert_uci_to_san(moves)[:move_num]
    legal_moves = [board.san(move) for move in list(board.legal_moves)]
    prompt = (
        f"Moves already made: {','.join(moves)}\n"
        f"Board Position with square names provided:\n{board_to_string_with_squares(board)}\n"
        f"Last move: {'Black' if board.turn else 'White'} played {last_move}, move_number {move_num} \n"
        f"Engine eval now: {convert_eval(current_eval, move_num % 2)}\n"
        f"Material balance (centipawn units, positive favors white): {100 * material_balance_current}\n"
        f"Engine classification of the last move quality: {move_quality}\n"
        f"Available moves: {','.join(legal_moves)}"
    )
    return prompt

def count_material_position(board):
    # Mapping chess.<piece> to labels
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    white_material = np.zeros(6, dtype=int)  # [pawn, knight, bishops, rooks, queens, total]
    black_material = np.zeros(6, dtype=int)

    for piece_type, value in piece_values.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        
        white_material[piece_type - 1] = white_count
        black_material[piece_type - 1] = black_count

        white_material[5] += white_count * value
        black_material[5] += black_count * value

    return np.concatenate((white_material, black_material))

def engine_eval_to_probability(eval, k=0.0025):
    """Convert a engine evaluation to a winning probability using a logistic function."""
    if eval > 900000:
        return 1.0
    elif eval < -900000:
        return 0.0
    return 1 / (1 + np.exp(-k * eval))

def quality_classification_thresholds(player_rating):
    # Adjust the logistic steepness based on player rating

    if player_rating > 2400:
        k = 0.0035  # Steeper for higher-rated players
    elif player_rating > 2000:
        k = 0.0030
    else:
        k = 0.0025  # Less steep for lower-rated players

    # Adjust thresholds - higher-rated players are expected to make fewer mistakes
    if player_rating > 2400:
        thresholds = {
            0: (0.00, 0.00),
            1: (0.00, 0.01),
            2: (0.01, 0.03),
            3: (0.03, 0.07),
            4: (0.07, 0.15),
            5: (0.15, 1.00)
        }
    elif player_rating > 2000:
        thresholds = {
            0: (0.00, 0.00),
            1: (0.00, 0.02),
            2: (0.02, 0.04),
            3: (0.04, 0.08),
            4: (0.08, 0.17),
            5: (0.17, 1.00)
        }
    else:
        thresholds = {
            0: (0.00, 0.00),
            1: (0.00, 0.02),
            2: (0.02, 0.05),
            3: (0.05, 0.10),
            4: (0.10, 0.20),
            5: (0.20, 1.00)
        }

    return k, thresholds

def classify_single_move_quality(move_prob, best_move_prob, previous_move_classification, k, thresholds):
    # Calculate expected points change
    expected_points_change = max(0, best_move_prob - move_prob)

    # Classify based on thresholds
    for classification, (lower, upper) in thresholds.items():
        if lower <= expected_points_change <= upper:
            move_classification = classification
            break

    # Detecting a "miss"
    if previous_move_classification in [4, 5] and move_classification in [4, 5]:
        move_classification = 6
    
    return move_classification

def classify_move_quality(game_moves, evaluations, top_move_evals, player_rating=2500, N=5, k=0.25):
    quality_values = {0: "best", 1: "excellent", 2: "good", 3: "inaccuracy", 4: "mistake", 5: "blunder", 6: "miss"}
    
    k, thresholds = quality_classification_thresholds(player_rating)

    classifications = []
    top_move_classifications = []

    previous_move_classification = None

    for i, move in enumerate(game_moves):

        move_prob = engine_eval_to_probability(evaluations[i], k)
        best_move_prob = engine_eval_to_probability(top_move_evals[i], k) if top_move_evals[i] else move_prob

        move_classification = classify_single_move_quality(move_prob, best_move_prob, previous_move_classification, k, thresholds)

        classifications.append(move_classification)

        previous_move_classification = move_classification

    return [quality_values[i] for i in classifications]

def convert_eval(move_eval, turn, mate_eval=1000000):
    if isinstance(move_eval, chess.engine.PovScore):
        if move_eval.is_mate():
            # Adjust the mate score representation
            if move_eval.mate() > 0:  # Checkmate for the current side
                move_eval = 1000000 - move_eval.mate()
            else:  # Getting checkmated
                move_eval = -1000000 - move_eval.mate()
        else:
            move_eval = move_eval.relative.score()

    move_eval = move_eval if turn != 0 else -move_eval
    if move_eval > mate_eval - 100:
        if move_eval == mate_eval:
            move_eval = 'Checkmate'
        else:
            move_eval = f"Mate in {mate_eval - move_eval}"
    else:
        move_eval = str(move_eval)
    return move_eval