# Chess Chat - A Chess Chatbot

![Chess Analysis GUI Demo](example.gif)

## Description

Chess Analysis GUI is a Python-based application that aims to revolutionize game review and analysis in chess. This chess chatbot allows users to review their games, similar to the chess.com game review feature, and interact live with a powerful Language Model (LLM). It combines chess engine analysis and Python chess utility functions with user queries to assist the LLM in providing meaningful insights into chess positions.

## Project Goal

The primary goal of this project is to create an interactive platform where chess enthusiasts can analyze their games and understand complex chess positions through an intuitive chat interface.

## Current State and Future Directions

As of now, the application uses OpenAI's GPT models for the chatbot functionality. While GPT models are powerful in various domains, their current capability in deep chess reasoning is limited. This project serves as a starting point, showcasing the basic concept of a chess-chatbot. Future developments may involve fine-tuning open-source LLMs like Llama2, specifically for chess position comprehension and analysis.

## Features

- Visual representation of chess games from URLs.
- Live interaction with a chatbot for game analysis and position explanation.
- Integration of chess engine analysis to enhance LLM responses.
- User-friendly GUI for seamless game navigation and interaction.

## Installation

### Prerequisites

- Python 3.x
- [Stockfish](https://stockfishchess.org/download/) or [LC0 (Leela Chess Zero)](https://lczero.org/play/download/)
- OpenAI API Key
- PyQt5
- chess (Python library)
- numpy

### Setup

1. Clone the repository:
    ```
    git clone https://github.com/sveinnpalsson/ChessChat
    ```
2. Install required Python packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

Run the main Python script to start the application:
