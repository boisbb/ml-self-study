import time

# https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python/

class TicTacToe:

    def __init__(self):
        self.current_state = [
                        ['.', '.', '.'],
                        ['.', '.', '.'],
                        ['.', '.', '.']]
        self.player_turn = 'X'
    
    def draw_board(self):
        for i in range(3):
            print('|', end='')
            for j in range(3):
                print(f' {self.current_state[i][j]} |', end='')
            print()
        print()
    
    def is_valid(self, px, py):
        if px < 0 or px > 2 or py < 0 or py > 2:
            return False
        elif self.current_state[px][py] != '.':
            return False
        return True

    def is_end(self):

        # Player won in vertical direction
        for i in range(3):
            if self.current_state[0][i] != '.':
                if self.current_state[0][i] == self.current_state[1][i]:
                    if self.current_state[1][i] == self.current_state[2][i]:
                        return self.current_state[0][i]
        
        # Player won in horizontal direction
        for i in range(3):
            if self.current_state[i] == ['X','X','X']:
                return 'X' 
            elif self.current_state[i] == ['O','O','O']:
                return 'O'
        
        # Player won in a main diagonal way
        if self.current_state[0][0] != '.':
            if self.current_state[0][0] == self.current_state[1][1]:
                if self.current_state[1][1] == self.current_state[2][2]:
                    return self.current_state[2][2]
        
        # Player won in a secondary diagonal way
        if self.current_state[0][2] != '.':
            if self.current_state[0][2] == self.current_state[1][1]:
                if self.current_state[1][1] == self.current_state[2][0]:
                    return self.current_state[2][0]
        
        # Full board
        for i in range(3):
            for j in range(3):
                if self.current_state[i][j] == '.':
                    return None
        
        return '.'

    def max_a_b(self, alpha, beta):
        maxv = -2
        px = None
        py = None

        result = self.is_end()

        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)
        
        for i in range(3):
            for j in range(3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'O'
                    (minv, min_i, min_j) = self.min_a_b(alpha, beta)

                    if minv > maxv:
                        maxv = minv
                        px = i
                        py = j
                    
                    self.current_state[i][j] = '.'

                    if maxv >= beta:
                        return (maxv, px, py)
                    
                    if maxv > alpha:
                        alpha = maxv
        return (maxv, px, py)
    
    def min_a_b(self, alpha, beta):
        minv = 2

        qx = None
        qy = None

        result = self.is_end()

        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)

        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    self.current_state[i][j] = 'X'
                    (m, max_i, max_j) = self.max_a_b(alpha, beta)
                    if m < minv:
                        minv = m
                        qx = i
                        qy = j
                    self.current_state[i][j] = '.'

                    if minv <= alpha:
                        return (minv, qx, qy)

                    if minv < beta:
                        beta = minv

        return (minv, qx, qy)


    def play(self):
        while True:
            self.draw_board()
            self.result = self.is_end()

            if self.result != None:
                if self.result == 'X':
                    print('The winner is X!')
                elif self.result == 'O':
                    print('The winner is O!')
                elif self.result == '.':
                    print("It's a tie!")
                return
            
            if self.player_turn == 'X':
                while True:
                    start = time.time()
                    (m, qx, qy) = self.min_a_b(-2, 2)
                    end = time.time()

                    print(f'Eval time: {round(end - start)}s')
                    print(f'Recommended move: X = {qx}, Y = {qy}')

                    px = int(input('Insert X coord of your move: '))
                    py = int(input('Insert Y coord of your move: '))

                    qx = px
                    qy = py

                    if self.is_valid(px, py):
                        self.current_state[px][py] = 'X'
                        self.player_turn = 'O'
                        break
                    else:
                        print('Invalid move, try again...')
            else:
                (m, px, py) = self.max_a_b(-2, 2)
                print(self.is_valid(px, py))
                self.current_state[px][py] = 'O'
                self.player_turn = 'X'


        