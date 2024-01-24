from argparse import ArgumentParser
import sys

from PyQt5.QtWidgets import QApplication
from gui_utils import ChessGUI
from chess_utils import fetch_game_moves

def main():
    parser = ArgumentParser(description="Chess Analysis GUI")
    parser.add_argument("--model_name", type=str, choices=["gpt3.5", "gpt4"], default="gpt4",
                        help="The model name for the OpenAI API. Choices: gpt35, gpt4")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="The maximum number of tokens for GPT completions.")
    parser.add_argument("--max_history_length", type=int, default=5,
                        help="Controls context length by setting a cap on chat history provided to the call to ChatGPT")
    parser.add_argument("--engine_name", type=str, choices=["stockfish", "lc0"], default="stockfish",
                        help="The name of the chess engine. Choices: stockfish, lc0")
    parser.add_argument("--engine_path", type=str, default=None,
                        help="The path of the chess engine executable. If leela chess zero (lc0) is used, then lc0_weights must also be provided")
    parser.add_argument("--lc0_weights", type=str, default=None)
    args = parser.parse_args()

    model_names = {
        "gpt3.5": "gpt-3.5-turbo",
        "gpt4": "gpt-4-1106-preview"
    }
    model_name_full = model_names[args.model_name]

    if args.engine_name == "lc0":
        if args.lc0_weights is None:
            raise ValueError("Weights file for lc0 must be provided (--lc0_weights)")
        engine_path = [args.engine_path, args.lc0_weights]
    else:
        engine_path = args.engine_path

    app = QApplication(sys.argv)
    initial_url = 'https://lichess.org/wI3YyUSi/black'
    initial_game_moves = fetch_game_moves(initial_url)
    gui = ChessGUI(model_name_full, args.engine_name, args.engine_path, initial_url=initial_url,
                   initial_game_moves=initial_game_moves, max_tokens=args.max_tokens,
                   max_history_length=args.max_history_length)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
