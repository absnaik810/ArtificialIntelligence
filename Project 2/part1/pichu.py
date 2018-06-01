#!/usr/bin/env python

import sys
import pdb
import copy
from time import sleep
import numpy as np
import os.path

#is kingfisher dead or not
def is_kingfisher_dead(board, player):
    dead = True
    king = "k"
    if player == "w":
        king = "K"

    for row in board:
        for  piece in row:
            if piece == king:
                dead = False
    return dead

'''
The code consists of individual successor functions for each of the different type of pieces.  To 
calculate the successor, we check if any of the boxes into which the coin can possibly move is empty,
or occupied by a coin of the opponent.  If yes, then we consider that box as a prospective target.   


For calculating the score we use two methods - the get_material_score() method and the 
get_mobility_score() method (Reference: https://chessprogramming.wikispaces.com/Evaluation).  

In the get_material_score() function, we calculate the total score on the basis of individual score
allocated to the coins.  For example, a KingFisher has more value compared to a Parakeet and so on.  
The get_mobility_score() function calculates the number of moves that a particular coin has.  Thus,
when the get_mobility_score() method is called for, say white, each of the white coins on the board is
looked at, and a score is calculated on the basis of number of moves each of the white coin has.

Besides, we consider uppercase characters to be representative of the White coins, while the lowercase
ones to be representative of the Black coins.   
'''
def get_mobility_score(board, turn):
    score = 0
    if turn == "w":
        for row in range (0, len(board)):
            for col in range (0, len(board[row])):
                if board[row][col] == "R":
                    score += len(get_robin_moves(board, [row ,col]))
                if board[row][col] == "N":
                    score += len(get_nighthawk_moves(board, [row, col]))
                if board[row][col] == "B":
                    score += len(get_bluejay_moves(board, [row, col])) 
                if board[row][col] == "Q":
                    len(get_quetzel_moves(board, [row, col]))
                if board[row][col] == "K":
                    len(get_kingfisher_moves(board, [row, col]))
                if board[row][col] == "P":
                    len(get_parakeet_moves(board, [row, col], turn))

    elif turn == "b":
         for row in range (0, len(board)):
            for col in range (0, len(board[row])):
                if board[row][col] == "r":
                    score += len(get_robin_moves(board, [row ,col]))
                if board[row][col] == "n":
                    score += len(get_nighthawk_moves(board, [row, col]))

                if board[row][col] == "b":
                    score += len(get_bluejay_moves(board, [row, col])) 
                if board[row][col] == "q":
                    len(get_quetzel_moves(board, [row, col]))
                if board[row][col] == "k":
                    len(get_kingfisher_moves(board, [row, col]))
                if board[row][col] == "p":
                    len(get_parakeet_moves(board, [row, col], turn))
                  
    return score

#gives material score for a board
def get_material_score(board):
    score = 0
    #send large neg value cuz whites king is dead
    if is_kingfisher_dead(board, "w"):
        return -1000
    #send large pos value cuz blacks king is dead
    if is_kingfisher_dead(board, "b"):
        return 1000
    
    for row in board:
        for piece in row:
            if piece.isupper():
                if piece == "P":
                    score += 5
                elif piece == "R":
                    score += 15
                elif piece == "B" or piece == "N":
                    score += 10
                elif piece == "Q":
                    score += 75
                elif piece == "K":
                    score += 100
            else:
                if piece == "p":
                    score += -5
                elif piece == "r":
                    score += -15
                elif piece == "b" or piece == "n":
                    score += -10
                elif piece == "q":
                    score += -75
                elif piece == "k":
                    score += -100
    return score

#converts the board to a printalbe 2D state for better visualization
def printable_board(board):
    res = []
    for row in board:
        res = res + row
    return "".join(res)
    
def printable_board2(board):
    return "\n".join([ " ".join([col for col in row]) for row in board])

#tests to see if unoccupied or occuped by w or b
def occupied_by(board, pos):
    return False if board[pos[0]][pos[1]] == "." else "w" if board[pos[0]][pos[1]].isupper() else "b"

#nighthawk generates all moves for nighthwak for current position
def get_nighthawk_moves(board, pos):
    typ = occupied_by(board, pos)
    #all the moves a nighthawk can make
    places = [ [-1,-2], [-2,-1], [-2,+1], [-1,+2], [+1, -2], [+2, -1], [+1,+2], [+2,+1] ]
    return [ [pos[0]+ p[0], pos[1]+p[1] ] for p in places if  0<=pos[0]+ p[0]<8 and 0<=pos[1]+p[1]<8\
            if typ != occupied_by(board, [pos[0]+ p[0],pos[1]+p[1]])]

#Kingfisher  generates all moves for kingfisher for current position
def get_kingfisher_moves(board, pos):
    typ = occupied_by(board, pos)
    #all moves the king can make
    places = [ [-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1],[1, 0],[1,1] ]
    return [ [pos[0]+ p[0], pos[1]+p[1] ] for p in places if  0<=pos[0]+ p[0]<8 and 0<=pos[1]+p[1]<8\
            if typ != occupied_by(board, [pos[0]+ p[0],pos[1]+p[1]]) ]

#bluejaymoves  generates all moves bluejay can make
#move in all 4 diagonal directions from current position
#and append if available or occupied by enemy
#then break
def get_bluejay_moves(board, pos):
    #get type fo piece
    typ = occupied_by(board, pos)
    move = []
    #upleft
    for x,y in zip(range(pos[0]-1, -1, -1),range(pos[1]-1, -1, -1)):
        curr_occupation  = occupied_by(board, [x,y])
        if not curr_occupation:
            move.append([x,y])
        elif curr_occupation != typ:
            move.append([x,y])
            break
        elif curr_occupation == typ:
            break 
    #upright
    for x,y in zip(range(pos[0]-1, -1, -1),range(pos[1]+1, 8, +1)):
        curr_occupation  = occupied_by(board, [x,y])
        if not curr_occupation:
            move.append([x,y])
        elif curr_occupation != typ:
            move.append([x,y])
            break
        elif curr_occupation == typ:
            break   
    #downleft
    for x,y in zip(range(pos[0]+1, 8),range(pos[1]-1, -1, -1)):
        curr_occupation = occupied_by(board, [x,y])
        if not curr_occupation:
            move.append([x,y])
        elif curr_occupation != typ:
            move.append([x,y])
            break
        elif curr_occupation == typ:
            break  
    #downright
    for x,y in zip(range(pos[0]+1, 8),range(pos[1]+1, 8)):
        curr_occupation = occupied_by(board, [x,y])
        if not curr_occupation:
            move.append([x,y])
        elif curr_occupation != typ:
            move.append([x,y])
            break
        elif curr_occupation == typ:
            break  
    return move

#ROOK  robin moves up down left right
#same a bluejay but move up down left right
def get_robin_moves(board, pos):
    typ = occupied_by(board, pos)
    moves = []
    #left 
    for i in range(pos[0]-1, -1, -1):
        curr_occupation = occupied_by(board, [i,pos[1]])
        if not curr_occupation:
            moves.append([i,pos[1]])
        elif curr_occupation != typ:
            moves.append([i,pos[1]])
            break
        elif curr_occupation == typ:
            break  

    #right
    for i in range(pos[0]+1, 8):
        curr_occupation = occupied_by(board, [i,pos[1]])
        if not curr_occupation:
            moves.append([i,pos[1]])
        elif curr_occupation != typ:
            moves.append([i,pos[1]])
            break
        elif curr_occupation == typ:
            break  

   #down
    for j in range(pos[1]+1, 8):
        curr_occupation = occupied_by(board, [pos[0],j])
        if not curr_occupation:
            moves.append([pos[0],j])
        elif curr_occupation != typ:
            moves.append([pos[0],j])
            break
        elif curr_occupation == typ:
            break  

   #up
    for j in range(pos[1]-1, -1, -1):
        curr_occupation = occupied_by(board, [pos[0],j])
        if not curr_occupation:
            moves.append([pos[0],j])
        elif curr_occupation != typ:
            moves.append([pos[0],j])
            break
        elif curr_occupation == typ:
            break  

    return moves 

#Quetzel Queen
#queen moves is sum of bluejay and robin
def get_quetzel_moves(board, pos):
    return get_robin_moves(board, pos) + get_bluejay_moves(board,pos)

#parakeet pawn
def get_parakeet_moves(board, pos, typ):
    typ = occupied_by(board, pos)
    moves = []
    if typ == "w":
        if pos[0]+1 < 8 and not occupied_by(board, [pos[0]+1, pos[1]] ):  
            moves.append([pos[0]+1 , pos[1]])
        
        #check for white to see if it is at its initial position then can do two moves 
        if pos[0] == 1 and  not occupied_by(board, [pos[0]+2, pos[1]] ):
            moves.append([pos[0]+2 , pos[1]])
        #check for enemy killing by pawn
        if pos[0]+1 <= 7 and  pos[1]-1 >= 0:
            if board[pos[0]+1][pos[1]-1] != "." and not board[pos[0]+1][pos[1]-1].isupper():
                moves.append([pos[0]+1 , pos[1]-1])
        
        if pos[0]+1 <= 7 and  pos[1]+1 <= 7: 
            if board[pos[0]+1][pos[1]+1] != "." and not board[pos[0]+1][pos[1]-1].isupper():
                moves.append([pos[0]+1 , pos[1]+1])   
    
        return moves
    
    elif typ == "b":
        if pos[0]-1 >= 0 and not occupied_by(board, [pos[0]-1, pos[1]] ):  
            moves.append([pos[0]-1 , pos[1]])
    
        if pos[0] == 6 and not occupied_by(board, [pos[0]-2, pos[1]] ):
            moves.append([pos[0]-2 , pos[1]])
        
        #check for enemy killing by pawn
        if pos[0]-1 >= 0  and pos[1]-1 >= 0:
            if board[pos[0]-1][pos[1]-1] != "." and board[pos[0]-1][pos[1]-1].isupper():
                moves.append([pos[0]-1 , pos[1]-1])
         
        if pos[0]-1 >= 0  and pos[1]+1 <= 7:
            if board[pos[0]-1][pos[1]+1] != "." and board[pos[0]-1][pos[1]+1].isupper():
                moves.append([pos[0]-1 , pos[1]+1])   
        
        return moves

#given a board piece and positon, it inserts the piece on that positon and returns the state
def insert_piece_on_board(board,piece, pos):
    return board[0:pos[0]] + [board[pos[0]][0:pos[1]] + [piece,] + board[pos[0]][pos[1]+1:]] + board[pos[0]+1:]

#given a pos/type of piece and board it generates all the successors for that piece
def get_move(board, pos):
    if board[pos[0]][pos[1]] == "N" or board[pos[0]][pos[1]] == "n":
        #store the type of piece
        piece = board[pos[0]][pos[1]]
        #variable to hold all successors for a piece
        moves = []
        #get all night hawk positons
        positions = get_nighthawk_moves(board, pos)
        #remove the piece from the current position
        tmp_board = insert_piece_on_board(board, ".", pos)
        for m in positions:
            moves.append(insert_piece_on_board(tmp_board,piece, m))

        return moves


    elif board[pos[0]][pos[1]] == "K" or board[pos[0]][pos[1]] == "k":
	    #store the type of piece
        piece = board[pos[0]][pos[1]]
        #variable to hold all successors for a piece
        moves = []
        #get all king fisher positons
        positions = get_kingfisher_moves(board, pos)
        #remove the piece from the current position
        tmp_board = insert_piece_on_board(board,".", pos)
        for m in positions:
            moves.append(insert_piece_on_board(tmp_board,piece, m))

        return moves    
        
    elif board[pos[0]][pos[1]] == "Q" or board[pos[0]][pos[1]] == "q":
    	#store the type of piece
        piece = board[pos[0]][pos[1]]
        #variable to hold all successors for a piece
        moves = []
        #get all quetzel positons
        positions = get_quetzel_moves(board, pos)
        #remove the piece from the current position
        tmp_board = insert_piece_on_board(board,".", pos)
        for m in positions:
            moves.append(insert_piece_on_board(tmp_board,piece, m))

        return moves   

    elif board[pos[0]][pos[1]] == "B" or board[pos[0]][pos[1]] == "b":
     	#store the type of piece
        piece = board[pos[0]][pos[1]]
        #variable to hold all successors for a piece
        moves = []
        #get all bluejay positons
        positions = get_bluejay_moves(board, pos)
        #remove the piece from the current position
        tmp_board = insert_piece_on_board(board,".", pos)
        for m in positions:
            moves.append(insert_piece_on_board(tmp_board,piece, m))

        return moves   

   		
    elif board[pos[0]][pos[1]] == "R" or board[pos[0]][pos[1]] == "r":
      	#store the type of piece
        piece = board[pos[0]][pos[1]]
        #variable to hold all successors for a piece
        moves = []
        #get all bluejay positons
        positions = get_robin_moves(board, pos)
        #remove the piece from the current position
        tmp_board = insert_piece_on_board(board,".", pos)
        for m in positions:
            moves.append(insert_piece_on_board(tmp_board,piece, m))

        return moves   
    
    # possible positions for parakreet
    elif board[pos[0]][pos[1]] == "P" or board[pos[0]][pos[1]] == "p":
        #store the type of piece
        piece = board[pos[0]][pos[1]]
        #variable to hold all successors for a piece
        moves = []
        #variable for positons of parakreet
        positions = []
        #get all bluejay positons
        if board[pos[0]][pos[1]].isupper():
            positions =  get_parakeet_moves(board, pos, "w")
        else:
    		positions =  get_parakeet_moves(board, pos, "b")
        #remove the piece from the current position
        tmp_board = insert_piece_on_board(board,".", pos)
        for m in positions:
            if m[0] == 0 and not piece.isupper() :
                moves.append(insert_piece_on_board(tmp_board,"q", m))
            elif m[0] == 7 and piece.isupper():
                moves.append(insert_piece_on_board(tmp_board,"Q", m))
            else:
                moves.append(insert_piece_on_board(tmp_board,piece, m))

        return moves   



#give the board and turn("w" or "b") it returns all the successors for all pieces on board
def get_successor(board, turn):
    succ = []
    for i in range(0,8):
        for j in range(0,8):
            #if it is white turn then get only white successors
            if turn == "w" and board[i][j] != "." and board[i][j].isupper():
                succ = succ + get_move(board, [i,j] )
            #if it is black turn then get only black successors
            elif turn == "b" and board[i][j] != "." and not board[i][j].isupper():
                succ = succ + get_move(board, [i,j] )
    return succ



def convert_board_to_tuple(board):
    res = []
    for row in board:
        res.append(tuple(row))

    return tuple(res)

#the pseudo code for minimax algorithm was referred from:
#https://en.wikipedia.org/wiki/Minimax
def minimax(board, depth, turn, alpha, beta):
    global move_dict
    global curr_board

    best_move = None 

    if depth == 0:
        return get_material_score(board) + 1.0*get_mobility_score(board, turn), best_move

    succ = get_successor(board, turn)
    
    if succ == []: return get_material_score(board) + -1.0* get_mobility_score(board, turn), best_move
    
    if turn  == "w":
        best_val = float('-inf')
        for s in succ:
            s_key = convert_board_to_tuple(s)
            if s_key in move_dict: 
                val = move_dict[s_key]
                if board == curr_board:
                    print "Current best move is"
                    print printable_board(s)

            else:
                val, move = minimax(s, depth-1, "b", alpha, beta)
                move_dict[s_key] = val
            if val > best_val:
                best_val = val
                best_move = s
                if board == curr_board:
                    print "Current best move is"
                    print printable_board(s)
            
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break

       
        return best_val, best_move
    
    if turn == "b":
        best_val = float('inf')
        for s in succ:
            s_key =  convert_board_to_tuple(s)
            if s_key in move_dict:
                val = move_dict[s_key]
                if board == curr_board:
                    print "Current best move is"
                    print printable_board(s)
 
            else:
                val, move = minimax(s, depth-1, "w", alpha, beta)
                move_dict[s_key] = val

            if val < best_val:
                best_val = val
                best_move = s
                if board == curr_board:
                    print "Current best move is"
                    print printable_board(s)

            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move


# white is capital and black for non-capital 
turn = sys.argv[1]
board_state = list(sys.argv[2])
time = int(sys.argv[3])
curr_board = [board_state[i:i+8] for i in range(0, len(board_state),8)]


# we are going to convert board to a tuple and save it in a file.
# so that we dont have to do that computation again
# to save dictonary to file I used the following link
#https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

#The idea of saving state came from here
#https://chessprogramming.wikispaces.com/Bitboards
#our idea is different as in we store board state and values
#for that state
#Since our program is simple and we run it again and again
#to get one move it is better to save some states 
#so that subsequent runs are faster

#move_dictonary is used to save already looked at states
#This is then subsequently used to lookup the moves 
#if move is found we just update the value

#not using the move dictonary for now as it gets
#really big in size and takes a lot of time to load
#last recorded size was 15MB
move_dict = {}
if os.path.exists("moves.npy") and False:
    move_dict = np.load('moves.npy').item()

best_val, best_move =  minimax(curr_board, 4, turn, float('-inf'), float('inf'))


#print win condition if black or white kingfisher is dead
if is_kingfisher_dead(best_move, "w"):
    print "Black wins"
#send large pos value cuz blacks king is dead
if is_kingfisher_dead(best_move, "b"):
    print "white wins"


print "The best suggested move with a value of ", best_val, "is "
print printable_board(best_move)

#np.save('moves.npy', move_dict) 


