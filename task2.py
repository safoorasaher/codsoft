import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Represents the 3x3 board
        self.current_winner = None  # Keeps track of the winner

    def print_board(self):
        for row in [self.board[i * 3:(i + 1) * 3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        # 0 | 1 | 2 etc (tells us what number corresponds to what box)
        number_board = [[str(i) for i in range(j * 3, (j + 1) * 3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # Check the row
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        # Check the column
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        # Check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]  # Left to right diagonal
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]  # Right to left diagonal
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

def minimax(position, maximizer_letter, minimizer_letter, max_turn, alpha, beta):
    if position.current_winner == minimizer_letter:
        return {'position': None, 'score': -1}
    elif position.current_winner == maximizer_letter:
        return {'position': None, 'score': 1}
    elif not position.empty_squares():
        return {'position': None, 'score': 0}

    if max_turn:
        max_score = {'position': None, 'score': -math.inf}
        for possible_move in position.available_moves():
            position.make_move(possible_move, maximizer_letter)
            sim_score = minimax(position, maximizer_letter, minimizer_letter, False, alpha, beta)
            position.board[possible_move] = ' '  # Undo the move
            sim_score['position'] = possible_move
            if sim_score['score'] > max_score['score']:
                max_score = sim_score
            alpha = max(alpha, sim_score['score'])
            if alpha >= beta:
                break
        return max_score
    else:
        min_score = {'position': None, 'score': math.inf}
        for possible_move in position.available_moves():
            position.make_move(possible_move, minimizer_letter)
            sim_score = minimax(position, maximizer_letter, minimizer_letter, True, alpha, beta)
            position.board[possible_move] = ' '  # Undo the move
            sim_score['position'] = possible_move
            if sim_score['score'] < min_score['score']:
                min_score = sim_score
            beta = min(beta, sim_score['score'])
            if alpha >= beta:
                break
        return min_score

def play_game():
    while True:
        game = TicTacToe()
        game.print_board_nums()  # Show the user what the numbers of the boxes are
        letter = 'X'  # Choose starting letter

        while game.empty_squares():
            if letter == 'X':
                square = None
                while square is None:
                    move = input("Enter position 0-8: ")
                    try:
                        square = int(move)
                        if square not in game.available_moves():
                            raise ValueError
                    except ValueError:
                        print("Invalid move. Please try again.")
                game.make_move(square, letter)
                game.print_board()  # Print the updated board after the player's move

            if game.current_winner:
                print(f"{game.current_winner} wins!")
                break  # Exit the inner loop when the game ends

            if letter == 'O':
                square = minimax(game, 'O', 'X', True, -math.inf, math.inf)['position']
                game.make_move(square, letter)
                game.print_board()  # Print the updated board after the AI's move

            if game.current_winner:
                print(f"{game.current_winner} wins!")
                break  # Exit the inner loop when the game ends

            letter = 'O' if letter == 'X' else 'X'  # Switch player

        if not input("Play again? (y/n): ").lower().startswith('y'):
            break  # Exit the outer loop if the user doesn't want to play again

if __name__ == "__main__":
    play_game()
