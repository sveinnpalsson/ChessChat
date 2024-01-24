import chess
import chess.engine
import chess.pgn
import chess.svg
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtGui import QFont, QTextCursor, QTextCharFormat, QColor
from PyQt5.QtSvg import QSvgWidget
import numpy as np
import re
import requests
import io
from copy import deepcopy
import json

from chess_utils import fetch_game_moves, analyze_game, analyze_position, count_material_position, classify_move_quality, board_summary
from gpt_utils import askgpt, cost

class ChessGUI(QWidget):
    def __init__(self, model_name, engine_name, engine_path, max_tokens=None, initial_url=None, initial_game_moves=None, max_history_length=None):
        super().__init__()
        self.setWindowTitle('Chess Engine Chat')
        
        # Initialize class variables
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.engine_name = engine_name
        self.engine_path = engine_path
        self.max_history_length = max_history_length
        self.total_cost = 0.0
        self.chat_msgs = []
        self.move_index = 0
        self.last_move_san = None
        self.move_evals = None
        self.top_move_evals = None
        self.move_quality = None
        self.board = chess.Board()
        self.game_moves = initial_game_moves if initial_game_moves else fetch_game_moves(initial_url)

        # GUI setup
        self.setup_gui()
        if initial_url is not None:
            self.url_input.setText(initial_url)
            self.on_url_enter()
            last_move_san = update_board_to_move(self.board, self.svg_widget, self.game_moves, 0)  # Initial board setup

    def setup_gui(self):
        font = QFont()
        font.setPointSize(13)

        main_layout = QHBoxLayout(self)
        board_layout = QVBoxLayout()
        main_layout.addLayout(board_layout)

        self.setup_url_input(board_layout, font)
        self.setup_svg_widget(board_layout)
        self.setup_move_buttons(board_layout, font)
        self.setup_chat_input(board_layout, font)
        self.setup_info_section(main_layout, font)
        self.connect_signals_to_slots()

    def connect_signals_to_slots(self):
        self.url_input.returnPressed.connect(self.on_url_enter)
        self.chat_input.returnPressed.connect(self.on_input_enter)
        self.prev_button.clicked.connect(self.on_prev_button_clicked)
        self.next_button.clicked.connect(self.on_next_button_clicked)
        self.move_number_input.returnPressed.connect(self.on_move_input)

    def setup_url_input(self, layout, font):
        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText("Enter game URL")
        self.url_input.setFont(font)
        layout.addWidget(self.url_input)

    def setup_svg_widget(self, layout):
        self.svg_widget = QSvgWidget(self)
        layout.addWidget(self.svg_widget)

    def setup_move_buttons(self, layout, font):
        # Create a layout for the move controls
        move_controls_layout = QHBoxLayout()

        # Previous move button
        self.prev_button = QPushButton("<", self)
        self.prev_button.setFont(font)
        move_controls_layout.addWidget(self.prev_button)

        # Input for move number
        self.move_number_input = QLineEdit(self)
        self.move_number_input.setPlaceholderText("Move number")
        self.move_number_input.setFont(font)
        move_controls_layout.addWidget(self.move_number_input)

        # Next move button
        self.next_button = QPushButton(">", self)
        self.next_button.setFont(font)
        move_controls_layout.addWidget(self.next_button)

        # Add the move controls layout to the parent layout
        layout.addLayout(move_controls_layout)

    def setup_chat_input(self, layout, font):
        self.chat_input = QLineEdit(self)
        self.chat_input.setPlaceholderText("Chat with ChessGPT")
        self.chat_input.setFont(font)
        layout.addWidget(self.chat_input)

    def setup_info_section(self, layout, font):
        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)
        self.info_text.setFont(font)
        layout.addWidget(self.info_text)

    def evaluate_move(self, move: str):
        """
        Given a move (san), returns a list of length 3 where each element 
        is a dict corresponding to the top 3 moves (best move first) 
        of the form {move: move, eval: eval_of_move, line: <list of moves (engine follow-up line)>}
        """
        board_next = deepcopy(self.board)
        if self.board.parse_san(move) not in self.board.legal_moves:
            return None

        board_next.push(self.board.parse_san(move))
        top_moves_info = analyze_position(board_next, self.engine_name, top_n=3)
        top_move_lines = []
        for i in range(len(top_moves_info)):
            board_tmp = deepcopy(board_next)
            line = []
            pv = top_moves_info[i]['pv']
            top_move_i = board_tmp.san(pv[0])
            board_tmp.push(pv[0])
            for j in range(1, min(4, len(pv))):
                line.append(board_tmp.san(pv[j]))
                board_tmp.push(pv[j])
            
            top_move_lines.append({
                "move": top_move_i, 
                "eval": convert_eval(top_moves_info[i]['score'], int(not self.board.turn)), 
                "line": line
            })
        return top_move_lines

    def on_input_enter(self):
        # User message
        user_text = f"User: {self.chat_input.text()}"
        self.info_text.append(user_text)
        user_start_pos = self.info_text.document().characterCount() - len(user_text) - 1
        user_end_pos = user_text.find(':') + 1 + user_start_pos
        highlight_text(self.info_text, user_start_pos, user_end_pos, 'red')
        QApplication.processEvents()

        # Construct prompt to ChatGPT
        material_balance = -np.diff(count_material_position(self.board)[np.array([5, 11])])[0]
        chess_prompt = board_summary(self.board, self.last_move_san, self.move_index, self.move_evals[self.move_index], material_balance, self.move_quality[self.move_index], self.game_moves)
        print(chess_prompt)

        # Temporary system message
        loading_text = "ChessGPT: Generating response..."
        self.info_text.append(loading_text)
        loading_start_pos = self.info_text.document().characterCount() - len(loading_text) - 1
        loading_end_pos = loading_start_pos + len(loading_text)
        highlight_text(self.info_text, loading_start_pos, loading_end_pos, 'green')
        QApplication.processEvents()

        # Call ChatGPT
        gpt_result, chat_history = askgpt(self.chat_input.text(), self.model_name, chess_prompt, msgs=self.chat_msgs, max_tokens=self.max_tokens)
        self.chat_msgs = chat_history[-self.max_history_length:] if self.max_history_length is not None else chat_history
        self.total_cost += cost(gpt_result, self.model_name)

        # Handle function call response
        if gpt_result.choices[0].finish_reason == 'function_call':
            fc = gpt_result.choices[0].message.function_call
            if fc.name == 'evaluate_move':
                top_lines = self.evaluate_move(json.loads(fc.arguments)["move"])
                print("---- Analyzing top lines ----")
                if top_lines is None:
                    self.info_text.append(f"ChessGPT: The move {json.loads(fc.arguments)['move']} is not a legal move in this position")
                    print(f"total cost: {self.total_cost}$")
                    return
                new_prompt = str(top_lines)
                gpt_result = askgpt("", new_prompt, msgs=self.chat_msgs, functions=[schema(self.evaluate_move)])
                self.total_cost += cost(gpt_result)

        # Print the response with HTML formatting
        gpt_text = f"ChessGPT: {gpt_result.choices[0].message.content}"
        gpt_text_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", gpt_text.replace('\n', '<br>'))
        gpt_text_html = gpt_text_html.replace("ChessGPT:", '<span style="color: blue;">ChessGPT:</span>', 1)

        # Append the formatted HTML content
        current_text = self.info_text.toHtml()
        self.info_text.setHtml(current_text + gpt_text_html)

        QApplication.processEvents()
        self.chat_input.clear()
        print(f"total cost: {self.total_cost}$")


    def on_url_enter(self):
        url = self.url_input.text()
        self.game_moves = fetch_game_moves(url)
        update_board_to_move(self.board, self.svg_widget, self.game_moves, 0)
        self.info_text.clear()
        self.info_text.append(f"New URL entered\nAnalyzing game...")
        QApplication.processEvents()

        game_analysis = analyze_game(self.game_moves, self.engine_name, self.engine_path)
        self.top_move_evals = [max(i['top_moves'].values()) for i in game_analysis]
        self.move_evals = [i['evaluation'] for i in game_analysis]
        self.move_quality = classify_move_quality(self.game_moves, self.move_evals, self.top_move_evals)
        self.info_text.append(f"Finished game analysis")

    def on_prev_button_clicked(self):
        if self.move_index > 0:
            self.move_index -= 1
            self.update_move(self.move_index)

    def on_next_button_clicked(self):
        if self.move_index < len(self.game_moves):
            self.move_index += 1
            self.update_move(self.move_index)
        
    def on_move_input(self):
        self.move_index = int(self.move_number_input.text())
        self.update_move(self.move_index)
    
    def update_move(self, move_index):
        if 0 <= move_index <= len(self.game_moves):
            self.last_move_san = update_board_to_move(self.board, self.svg_widget, self.game_moves, move_index)
            if move_index > 0:
                self.info_text.append(f"Move {move_index}: {self.last_move_san}")
            elif move_index == 0:
                self.info_text.append(f"Initial position")
            self.move_number_input.setText(str(move_index))


def highlight_text(text_edit, start_pos, end_pos, color):
    cursor = QTextCursor(text_edit.document())
    cursor.setPosition(start_pos, QTextCursor.MoveAnchor)
    cursor.setPosition(end_pos, QTextCursor.KeepAnchor)

    format = QTextCharFormat()
    format.setForeground(QColor(color))
    cursor.mergeCharFormat(format)


def update_board_to_move(board, svg_widget, moves, move_number):
    board.reset()
    move_ = None
    move_san = None
    for move in moves[:move_number]:
        try:
            move_ = board.parse_san(move)
            move_san = board.san(move)
            board.push(move_)
        except:
            move_ = chess.Move.from_uci(move)
            move_san = board.san(move_)
            board.push(move_)
    
    board_svg = chess.svg.board(board, lastmove=move_).encode('UTF-8')
    svg_widget.load(board_svg)
    return move_san