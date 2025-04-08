from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import numpy as np
import time
import heapq
from collections import deque
from typing import List
import copy
# import multiprocessing

class PlayerController:
    # for the controller to read
    def __init__(self, time_left: Callable):
        self.map = None
        self.board_dim_x = None
        self.board_dim_y = None
        # self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.enemy_close_threshold = 12
        self.enemy_close_threshold_defense = 15
        self.player_head = None
        self.enemy_head = None
        self.player_enemy_distance = None
        self.player_enemy_distance_astar = None
        self.enemy_player_distance_astar = None
        self.player_enemy_path = None
        self.enemy_player_path = None
        self.length = None
        self.path_to_apple = []
        self.current_apple = None
        self.turn_count = 0
        self.time = 0
        return

    def bid(self, board:player_board.PlayerBoard, time_left:Callable):
        self.board_dim_x = board.get_dim_x()
        self.board_dim_y = board.get_dim_y()
        return 0
    

    def play(self, board:player_board.PlayerBoard, time_left:Callable):
        return self.play_empty(board, time_left)
    

    def find_enemy_distance(self, board:player_board.PlayerBoard):
        """
        Finds the distance between the player and the enemy while addressing portals
        """
        portal_dict = board.get_portal_dict()
        # We find the normal distance
        # Then we find the distance from head to portal and then portal destination to enemy to enemy
        # we do this for all portals
        # and then we take the minimum distance
        # Calculate the normal distance
        normal_distance = np.linalg.norm(np.array(self.player_head) - np.array(self.enemy_head))

        # Initialize the minimum distance with the normal distance
        min_distance = normal_distance

        # Iterate through all portals
        for portal_start, portal_end in portal_dict.items():
            # Distance from player head to the portal start
            distance_to_portal = np.linalg.norm(np.array(self.player_head) - np.array(portal_start))

            # Distance from portal end to the enemy head
            distance_from_portal_to_enemy = np.linalg.norm(np.array(portal_end) - np.array(self.enemy_head))

            # Total distance using the portal
            total_portal_distance = distance_to_portal + distance_from_portal_to_enemy

            # Update the minimum distance if the portal path is shorter
            min_distance = min(min_distance, total_portal_distance)

        return min_distance
    
    
    def find_enemy_distance_astar(self, board:player_board.PlayerBoard):
        self.player_enemy_path = self.astar_find_player(board, board.get_head_location(), board.get_head_location(enemy=True))
        self.player_enemy_distance_astar = len(self.player_enemy_path) if self.player_enemy_path is not None else -1
    
    def find_enemy_player_distance(self, board:player_board.PlayerBoard):
        self.enemy_player_path = self.astar_find_player(board, board.get_head_location(enemy=True), board.get_head_location(), enemy=True)
        self.enemy_player_distance_astar = len(self.enemy_player_path) if self.enemy_player_path is not None else -1

    
    def find_winning_turn_no_depth_iterative(self, board: player_board.PlayerBoard, enemy_1_move, max_depth=7):
        """
        Goes through all of our moves until 5 moves in one turn to see if any move wins
        @param: enemy_1_move: list of moves that lead to only one move for the enemy. This is used in find_winning_turn_multiple_moves_5_layers
        @return: the list of moves that lead to a win
        """
        queue = deque()
        queue.append((board, [], 1))
        # stack = [(board, [], 1)]  # Stack to simulate recursion: (board, moves, depth)

        while queue:
            self.i += 1
            current_board, current_moves, current_depth = queue.popleft()

            if current_depth > max_depth:
                # print(f"Exceeded max depth for moves: {current_moves}")
                continue

            # Get all possible moves
            valid_moves = self.valid_moves_list(current_board)

            for move in valid_moves:
                # Forecast the move
                new_board, success = current_board.forecast_move(move)
                if not success:
                    print(f"Failed to forecast move: {current_moves + [move]}")
                    continue

                # Check if the opponent has no valid moves
                opp_valid_moves = self.valid_moves_list(new_board, enemy=True)
                if not opp_valid_moves:
                    # Found a winning sequence
                    # Check if after our move, the opponent can lay traps to have more moves and after having more moves, do we have any move
                    new_board_copy: player_board.PlayerBoard = new_board.get_copy()
                    new_board_copy.end_turn()
                    new_board_length = new_board_copy.get_length(enemy=True)
                    play_move = True
                    while new_board_length > 2:
                        # Attempt to apply a trap
                        if new_board_copy.is_valid_trap(enemy=True):
                            new_board_copy.apply_trap(check_validity=False)
                        else:
                            break
                        # Update traps remaining and used
                        new_board_length = new_board_copy.get_length(enemy=True)

                    # Check if a safe move exists after applying the trap
                    opponent_moves = self.valid_moves_list(new_board_copy, enemy=True)
                    play_move = len(opponent_moves) == 0
                    if play_move:        
                        return current_moves + [move]
                if len(opp_valid_moves) == 1:
                    enemy_1_move.append(current_moves + [move])

                # Push the next state onto the stack
                queue.append((new_board, current_moves + [move], current_depth + 1))

        return None  # No winning move found

    def find_winning_turn_multiple_moves_5_layers(self, board: player_board.PlayerBoard, enemy_1_move, max_depth=5):
        """
        Goes through all moves that led to opponent having one move, plays opponent's move
        Opponent can play multiple moves in one turn, so we check how many moves available now
        If one move, we play it and add it to our stack/queue/call recusive function as well
        We do this until 3 moves in one turn
        Then we check all of our moves until 3 moves in one turn to see if they lead to a win
        We check if opponent has no moves in which case we return our move, if opponent does have a move, we repeat the above
        We does until 5 depth (our move, opponent move is one depth)
        @param enemy_1_move -  2d list of our moves that lead to opp having one move.
        @param max_depth - maximum in future we wanna look. 
        """
        return None


    def attack(self, board:player_board.PlayerBoard, time_left:Callable):
        if self.player_enemy_distance <= self.enemy_close_threshold:
            if self.player_enemy_distance_astar is None:
                self.find_enemy_distance_astar(board)
            if self.time < 100:
                number_of_playable_moves = min(9, (pow(((board.get_length() - 2) * 8) + 1, 0.5) - 1) / 2 + 1)
            else:
                number_of_playable_moves = (pow(((board.get_length() - 2) * 8) + 1, 0.5) - 1) / 2 + 1
            if self.player_enemy_distance_astar != -1 and self.player_enemy_distance_astar <= number_of_playable_moves:
                # We start with our first stage which goes through all of our moves up until 5 moves to see if any move wins
                # This stage can also save the moves that lead to only one move for our next stage
                max_depth = 7
                if self.time < 50:
                    max_depth = 4
                elif self.time < 100:
                    max_depth = 5
                self.i = 0
                board_astar, astar_moves = self.play_astar_moves_until_threshold(board)
                enemy_1_move = []
                moves = self.find_winning_turn_no_depth_iterative(board_astar, enemy_1_move, max_depth)
                print(self.i)
                if moves is not None:
                    print(f"Found winning move", moves)
                    return astar_moves + moves
                # if enemy_1_move:
                #     moves = self.find_winning_turn_multiple_moves_5_layers(board_astar, enemy_1_move, max_depth)
                #     if moves is not None:
                #         print(f"Found winning move", moves)
                #         return moves
                

    def play_empty(self, board:player_board.PlayerBoard, time_left:Callable):
        self.length = board.get_unqueued_length()
        self.player_head = board.get_head_location()
        self.enemy_head = board.get_head_location(enemy=True)
        self.turn_count = board.get_turn_count()
        # Check if the enemy is close to us
        self.player_enemy_distance = self.find_enemy_distance(board)
        self.player_enemy_distance_astar = None
        self.enemy_player_distance_astar = None
        self.player_enemy_path = None
        self.enemy_player_path = None
        self.time = time_left()

        winning_turn = self.attack(board, time_left)
        if winning_turn is not None:
            return winning_turn
            
        possible_moves = board.get_possible_directions()
        final_moves = [move for move in possible_moves if board.is_valid_move(move)]    
        head_location = board.get_head_location()
        apple_locations = board.get_current_apples()
        future_apples = board.get_future_apples()
        
        length_snake = board.get_length()
        closest_apples = self.get_closest_apple(head_location, apple_locations, future_apples, length_snake)
        # Gets the closest apple if there is an apple on the board, else just goes to an open space

        # if self.length > 30:
        #     # If we are long enough, we can start playing aggressively, go towards opponent
        #     final_moves = self.sort_towards_opponent(board, final_moves)
        #     final_moves = self.get_move_to_open_space(board, final_moves, ignore2=False)
        # else:  
        if closest_apples is not None:
            if (
                self.current_apple is None
                or self.path_to_apple is None or tuple(map(int, head_location)) not in self.path_to_apple
                or not any(np.array_equal(self.current_apple, apple) for apple in closest_apples[:3]) 
            ):
                paths = [self.astar(board, head_location, closest_apples[i]) for i in range(min(3, len(closest_apples)))]
                valid_paths = [(i, p) for i, p in enumerate(paths) if p is not None]
                # Check if there are any valid paths before calling min()
                if valid_paths:
                    min_index, path = min(valid_paths, key=lambda x: len(x[1]))
                    self.path_to_apple = path
                    self.current_apple = closest_apples[min_index]
                else:
                    min_index, path = None, [] 
                    self.path_to_apple = None
            
            final_moves.sort(key=lambda move: self.move_priority(board, move, closest_apples))
            if self.path_to_apple:
                try:
                    move_a_star = self.path_to_apple[tuple(map(int, head_location))]
                    if board.is_valid_move(move_a_star):
                        final_moves.insert(0, move_a_star)
                except:
                    print("error")
            final_moves = self.get_move_to_open_space(board, final_moves, ignore2=length_snake < 12)
        else:
            final_moves = self.get_move_to_open_space(board, final_moves, sort_all_open_space=True)
            
        
        final_moves = self.sort_move_to_edge(board, final_moves)
        final_turn = []

        # Final moves is a list of moves. We go through each move. If the move_is_safe returns some number of moves in a turn that we lose from, we store
        # that depth in a dictionary. If this function returns -1 and self.move_is_safe_10_moves also returns True, we break
        # If all of them have failed, we take multiple moves and then use has_safe_move to see if some move is safe
        # If that fails, we apply a trap and see if that works
        # If that fails, we return the move that has the highest depth
        moves_to_lose = {}
        i = 0
        while True:
            if i < len(final_moves):
                if self.player_enemy_distance <= self.enemy_close_threshold_defense:
                    new_board, _ = board.forecast_turn(final_moves[i], check_validity=False)
                    print(f"Move we forecasted {final_moves[i]}")
                    self.find_enemy_player_distance(new_board)
                    if self.time < 100:
                        number_of_playable_moves = min(9, (pow(((new_board.get_length(enemy=True) - 2) * 8) + 1, 0.5) - 1) / 2 + 1)
                    else:
                        number_of_playable_moves = (pow(((new_board.get_length(enemy=True) - 2) * 8) + 1, 0.5) - 1) / 2 + 1
                    if (self.enemy_player_distance_astar != -1 and self.enemy_player_distance_astar < number_of_playable_moves): 
                        max_depth = 8
                        if self.time < 50:
                            max_depth = 4
                        elif self.time < 100:
                            max_depth = 5
                        one_move_list = []
                        board_astar, astar_moves = self.play_astar_moves_until_threshold(new_board, threshold=4, enemy=True)
                        number_of_moves = self.move_is_safe_one_layer(board_astar, one_move_list, max_depth)
                        if one_move_list:
                            if number_of_moves == -1 and  self.move_is_safe_5_layers_multiple_moves(board_astar, final_moves[i], one_move_list, max_depth) and self.move_is_safe_10_moves(board, final_moves[i]):
                                break
                        else:
                            if number_of_moves == -1 and self.move_is_safe_10_moves(board, final_moves[i]):
                                break
                        if number_of_moves != -1:
                            moves_to_lose[final_moves[i]] = number_of_moves + len(astar_moves)
                    else:
                        if self.move_is_safe_10_moves(board, final_moves[i]):
                            break
                else:
                    # print(f"not close, {self.player_enemy_distance}")
                    if self.move_is_safe_10_moves(board, final_moves[i]):
                        break
                print("not safe")
                i += 1
            else:
                print(f"except, {final_moves}, {i}")

                # Try multiple moves first
                multiple_moves = self.defense_multiple_moves(board, final_moves)
                if multiple_moves:
                    print("multiple moves: ", multiple_moves)
                    return multiple_moves
                
                i -= 1
                safe = False
                number_of_traps = 0
                # new_board = board.get_copy()
                new_board_length = length_snake
                while new_board_length > 2:
                    print("trap")
                    # Attempt to apply a trap
                    if board.is_valid_trap():
                        number_of_traps += 1
                        board.apply_trap(check_validity=False)
                        final_turn.append(Action.TRAP)
                    else:
                        print("trap failed")
                        break

                    new_board_length = board.get_length()

                    # Check if a safe move exists after applying the trap
                    if self.has_safe_move(board):
                        safe = True
                        print("safe move")
                        # for _ in range(number_of_traps):
                            # success = board.apply_trap()
                            # if success:
                            #     final_turn.append(Action.TRAP)
                        # apply traps and then call this function again to find a winning move or find the best move based on our other stuff
                        final_turn += self.play_empty(board, time_left)
                        return final_turn
                if not safe:
                    multiple_moves = self.defense_multiple_moves(board, final_moves)
                    if multiple_moves:
                        print("multiple moves: ", multiple_moves)
                        return multiple_moves
                    print("no safe move")
                    print(moves_to_lose)
                    if len(moves_to_lose) == 0:
                        i = 0
                        break
                    moves_to_lose = {move: moves_to_lose[move] for move in moves_to_lose if board.is_valid_move(move)}
                    max_moves = max(moves_to_lose.values())
                    for move in moves_to_lose:
                        if moves_to_lose[move] == max_moves:
                            final_turn.append(move)
                            break
                    return final_turn

        if i < len(final_moves) and board.apply_move(final_moves[i]):
            final_turn.append(final_moves[i])
        else:
            print("failed to apply move")
            return self.valid_moves_list(board)[0]

        return final_turn
    

    def defense_multiple_moves(self, board, final_moves):
        multiple_enemy_moves_list = [[]]
        print("trying multiple moves")
        for our_moves in multiple_enemy_moves_list:
            latest_board = board.get_copy()
            for moves in our_moves:
                latest_board.apply_move(moves)
            next_moves = self.valid_moves_list(latest_board) if our_moves else final_moves
            for move in next_moves:
                if self.check_move(board, our_moves + [move]):
                    return our_moves + [move]
                multiple_enemy_moves_list.append(our_moves + [move])
        return []


    def get_move_to_open_space_sort_helper(self, board, move, ignore2=False, sort_all_open_space=False):
        loc = board.get_loc_after_move(move)
        if board.has_enemy_trap(loc[0], loc[1]):
            return float('inf')
        new_board, status = board.forecast_move(move)
        possible_moves_forecast = new_board.get_possible_directions()
        final_moves_forecast = []
        for move in possible_moves_forecast:
            if new_board.is_valid_move(move):
                loc = new_board.get_loc_after_move(move)
                if not new_board.has_enemy_trap(loc[0], loc[1]):
                    final_moves_forecast.append(move)
        if len(final_moves_forecast) == 0:
            return float('inf')
        elif len(final_moves_forecast) == 1:
            return 10000
        elif not ignore2 and len(final_moves_forecast) == 2:
            return 1000
        if sort_all_open_space:
            return 0 - len(final_moves_forecast)
        return 0
    
    def get_move_to_open_space(self, board, possible_moves, ignore2=False, sort_all_open_space=False):
        return sorted(possible_moves, key=lambda move: self.get_move_to_open_space_sort_helper(board, move, ignore2, sort_all_open_space))

  
    def get_closest_apple_from_apples_helper(self, head_location, apple_loc):
        if self.player_enemy_distance > 10 and (apple_loc[0] == self.board_dim_x - 1 or apple_loc[1] == self.board_dim_y - 1 or apple_loc[0] == 0 or apple_loc[1] == 0):
            return float('inf')
        return self.distance_to_apple(head_location, apple_loc)
    
    def get_closest_apple_from_apples(self, head_location, apple_locations):
        return sorted(apple_locations, key=lambda apple: self.get_closest_apple_from_apples_helper(head_location, apple))
    
    def get_closest_apple(self, head_location, apple_locations, future_apples, snake_length):
        closest_apple = None
        if len(apple_locations) == 0:
            # pass
            if self.turn_count > 1000 and len(future_apples) != 0 and snake_length < 12:
                spawn_time = future_apples[0][0]
                index = 0
                for row in future_apples:
                    if row[0] != spawn_time:
                        break
                    index += 1
                same_time_apples = future_apples[:index, 1:]
                closest_apple = self.get_closest_apple_from_apples(head_location, same_time_apples)
        else:
            closest_apple = self.get_closest_apple_from_apples(head_location, apple_locations)
        return closest_apple
    
    def sort_move_to_edge_helper(self, board, move):
        new_location = board.get_loc_after_move(move)
        new_board, status = board.forecast_move(move)
        possible_moves_forecast = new_board.get_possible_directions()
        final_moves_forecast_no_traps = []
        final_moves_forecast = []
        for move in possible_moves_forecast:
            if new_board.is_valid_move(move):
                loc = new_board.get_loc_after_move(move)
                if not new_board.has_enemy_trap(loc[0], loc[1]):
                    final_moves_forecast_no_traps.append(move)
                final_moves_forecast.append(move)
        if len(final_moves_forecast) == 0:
            return float('inf')
        if board.has_enemy_trap(new_location[0], new_location[1]):
            return 11000
        if len(final_moves_forecast_no_traps) == 0:
            return 10010
        
        if new_location[0] == self.board_dim_x - 1 or new_location[1] == self.board_dim_y - 1 or new_location[0] == 0 or new_location[1] == 0:
            if (self.length < 4 or self.player_enemy_distance > 10) and board.has_apple(new_location[0], new_location[1]):
                return 0
            return 10000
        return 0
    
    def sort_move_to_edge(self, board, possible_moves):
        return sorted(possible_moves, key=lambda move: self.sort_move_to_edge_helper(board, move))


    def sort_towards_opponent(self, board, possible_moves):
        return sorted(possible_moves, key=lambda move: self.sort_move_towards_opponent_helper(board, move))
    
    def sort_move_towards_opponent_helper(self, board, move):
        new_location = board.get_loc_after_move(move)
        return np.linalg.norm(new_location - self.enemy_head)
    
    def move_priority(self, board, move, closest_apple):
        new_location = board.get_loc_after_move(move)
        return self.distance_to_apple(new_location, closest_apple)
    
    def distance_to_apple(self, location, apple_loc):
        return np.linalg.norm(location - apple_loc)
    

    def check_move(self, board: player_board.PlayerBoard, move, max_depth=0):
        new_board, _ = board.forecast_turn(move)
        self.find_enemy_player_distance(new_board)
        if not max_depth:
            max_depth = 8
            if self.time < 50:
                max_depth = 4
            elif self.time < 100:
                max_depth = 5
        if (self.enemy_player_distance_astar != -1 and self.enemy_player_distance_astar < self.enemy_close_threshold_defense): 
            one_move_list = []
            board_astar, _ = self.play_astar_moves_until_threshold(new_board, threshold=4, enemy=True)
            number_of_moves = self.move_is_safe_one_layer(board_astar, one_move_list, max_depth)
            if one_move_list:
                if number_of_moves == -1 and self.move_is_safe_5_layers_multiple_moves(board_astar, move, one_move_list, max_depth) and self.move_is_safe_10_moves(board, move):
                    return True
            else:
                if number_of_moves == -1 and self.move_is_safe_10_moves(board, move):
                    return True
        else:
            if self.move_is_safe_10_moves(board, move):
                return True
        return False

    def has_safe_move(self, board: player_board.PlayerBoard, print_moves=False) -> bool:
        """
        Checks if there exists a move for the current player such that,
        after playing that move, no opponent response can lead to a state where the player dies.
        Uses move_is_safe and move_is_safe_10_moves to determine safety.

        :param board: The current board state.
        :return: True if there exists a safe move, False otherwise.
        """
        # Get all possible moves for the current player
        valid_moves = self.valid_moves_list(board)

        # If no valid moves, return False immediately
        if not valid_moves:
            return False
        if print_moves:
            print(valid_moves)
        max_depth = 7
        if self.time < 50:
            max_depth = 4
        elif self.time < 100:
            max_depth = 5
        for move in valid_moves:
            if self.check_move(board, move, max_depth):
                break
        # If no move satisfies the condition, return False
        return False

    def move_is_safe_one_layer(self, board: player_board.PlayerBoard, one_move_list, max_moves_one_turn=6):
        """
        Plays your move, then checks if all of opponent's move until 5 moves in one turn lead to our death
        @return: -1 if no move leads to our death, otherwise the number of moves played in the same turn to lead to our death
        """
        return self.move_is_safe_one_layer_helper(board, one_move_list, max_moves_one_turn)
    
    def play_astar_moves_until_threshold(self, board: player_board.PlayerBoard, threshold = 2, enemy=False):
        new_board = board.get_copy()
        i = 0
        distance = self.player_enemy_distance_astar if not enemy else self.enemy_player_distance_astar
        path = self.player_enemy_path if not enemy else self.enemy_player_path
        moves = []
        while i + threshold < distance:
            head_location = new_board.get_head_location(enemy=enemy)
            move = path.get(tuple(map(int, head_location)))
            i += 1
            moves.append(move)
            success = new_board.apply_move(move)
            if not success:
                break
        return new_board, moves


    def move_is_safe_one_layer_helper(self, new_board: player_board.PlayerBoard, one_move_list, max_moves_one_turn=6):
        # Queue for BFS-like simulation of opponent's moves
        queue = deque()
        queue.append((new_board, [], 0))  # (board_state, depth)

        while queue:
            current_board, moves, depth = queue.popleft()

            # Get all opponent's possible moves at this depth
            enemy_valid_moves = self.valid_moves_list(current_board, enemy=True)

            for enemy_move in enemy_valid_moves:
                # Forecast the opponent's move
                opponent_board, _ = current_board.forecast_move(enemy_move, check_validity=False)

                # Check if we still have at least one valid move
                my_possible_moves = opponent_board.get_possible_directions()
                my_valid_moves = [
                    my_move for my_move in my_possible_moves if opponent_board.is_valid_move(my_move)
                ]

                # If we have no valid moves after this opponent's response, this path is unsafe
                if not my_valid_moves:
                    return depth  
                if len(my_valid_moves) == 1:
                    one_move_list.append(moves + [enemy_move])

                # Continue forecasting up to max_depth
                if depth < max_moves_one_turn:
                    queue.append((opponent_board, moves + [enemy_move], depth + 1))


        # If all opponent sequences were safe, return True
        return -1

    def move_is_safe_5_layers_multiple_moves(self, board: player_board.PlayerBoard, move_to_play, one_move_list, max_depth=5, max_enemy_moves=3):
        """
        Simulates our move, then applies all opponent moves that when played cause us to have 1 valid move. Repeats this process for up to 5 depth levels using iteration instead of recursion.
        Allows checking more than 3 enemy moves per turn without using nested loops. 
        paramters:
            board - current board
            move_to_play - move we want to play
            one_move_list - list of enemy moves that when played cuase us to have only 1 valid move
        Returns:
            0 - if a future move leads to no valid moves (death)
            1 - if after 5 depth we have only 1 move left
            2 - if we remain safe with multiple moves available
        # """
        print(f"beg: {move_to_play}, {one_move_list}")
        if not one_move_list:
            return 2
        print("started")
        stack = [([board], 1)]
        
        while stack:            
            curr_board_list, curr_depth = stack.pop()
            for curr_board in curr_board_list:
                next_board_list = []
                if curr_depth == 1:
                    curr_board.apply_turn(move_to_play)
                    next_board_list = [curr_board.forecast_turn(e_move)[0] for e_move in one_move_list]
                else:
                    valid_single_move = self.valid_moves_list(curr_board)
                    curr_board.apply_turn(valid_single_move)  

                    multiple_one_moves_list = [[]]
                
                    for enemy_moves in multiple_one_moves_list:
                        
                        if len(enemy_moves) == max_enemy_moves + 1:
                            break
                        
                        latest_board = curr_board.get_copy()
                        for e_move in enemy_moves:
                            latest_board.apply_move(e_move)
                        next_moves = self.valid_moves_list(latest_board, enemy=True)
                        for move in next_moves:
                            multiple_one_moves_list.append(enemy_moves + [move])
                            e_board = latest_board.forecast_turn(enemy_moves + [move])[0]
                            our_moves = self.valid_moves_list(e_board)
                        
                            if len(our_moves) == 0:
                                print(enemy_moves + [move])
                                return 0  # Death scenario
                            elif len(our_moves) == 1:
                                next_board_list.append(e_board)
                        
                if curr_depth < max_depth and next_board_list:
                    stack.append((next_board_list, curr_depth + 1))
                                    
            if curr_depth < max_depth and next_board_list:
                pass
            else:
                print(f"res {next_board_list}")
                return 1 if next_board_list else 2 
        
        return 2  # Default safe case


    def move_is_safe_10_moves(self, board: player_board.PlayerBoard, move):
        """
        Plays our move, then plays opponent's move that leads to our least number of moves, then plays our move
        Does this 10 times to see if our move is safe
        @return: True if our move is safe, False otherwise
        """
        return self.is_safe_recursive(board.forecast_turn(move, check_validity=False)[0], 10)

    def is_safe_recursive(self, board: player_board.PlayerBoard, depth):
        valid_moves_enemy = self.valid_moves_list(board, enemy=True)
        if len(valid_moves_enemy) == 0:
            board.end_turn()
        else:
            best_move = None
            min_moves = float('inf')
            for enemy_move in valid_moves_enemy:
                opponent_board, success = board.forecast_turn(enemy_move)
                if not success:
                    continue
                my_valid_moves = self.valid_moves_list(opponent_board)
                enemy_valid_moves = self.valid_moves_list(opponent_board, enemy=True)
                if len(enemy_valid_moves) != 0 and len(my_valid_moves) < min_moves:
                    min_moves = len(my_valid_moves)
                    best_move = enemy_move
            if best_move is None:
                board.end_turn()
            else:
                board.apply_turn([best_move], check_validity=False)
        if depth == 0:
            return bool(self.valid_moves_list(board))

        valid_moves = self.valid_moves_list(board)

        # if len(valid_moves) > 3:
        #     return True
        if len(valid_moves) == 0:
            return False

        for move in valid_moves:
            new_board, success = board.forecast_turn(move)
            if not success:
                continue
            if self.is_safe_recursive(new_board, depth - 1):
                return True

        return False
    
    def valid_moves_list(self, board, enemy=False):
        possible_moves = board.get_possible_directions(enemy=enemy)
        valid_moves = [move for move in possible_moves if board.is_valid_move(move, enemy=enemy)]
        return valid_moves

    def astar(self, grid: player_board.PlayerBoard, start, end, enemy=False):
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        
        row, col = grid.get_dim_x(), grid.get_dim_y()
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: pow(pow(start[0] - end[0], 2) + pow(start[1] - end[1], 2), 0.5)}
        
        directions = {(-1, 0): Action.WEST, (1, 0): Action.EAST, (0, -1): Action.NORTH, (0, 1): Action.SOUTH, (-1, -1): Action.NORTHWEST, (-1, 1): Action.SOUTHWEST, (1, -1): Action.NORTHEAST, (1, 1): Action.SOUTHEAST}
        
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            board = grid
            if current[0] == end[0] and current[1] == end[1]:
                # print(f"found at {current}")
                path = {}
                move_taken = None
                while current in came_from:
                    current, move_taken = came_from[current]
                    path[current] = move_taken

                return path


            if current in visited:
                continue
            
            visited.add(current)

            prev_current = current
            moves_taken = []
            while prev_current in came_from:
                moves_taken.append(came_from[prev_current][1])
                prev_current = came_from[prev_current][0]
            moves_taken.reverse()
            for m in moves_taken:
                board, _ = board.forecast_move(m, sacrifice=1)

            for d in directions:
                neighbor = (current[0] + d[0], current[1] + d[1])
                                
                if 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
                    move = directions[d]
                    
                    if board.is_portal(neighbor[0], neighbor[1]):
                        # print(f"portal found at {neighbor}")
                        neighbor = tuple(int(i) for i in board.get_portal_dest(neighbor[0], neighbor[1]))
                    
                    if (board.is_valid_move(move, sacrifice=1, enemy=enemy)):
                        tentative_g_score = g_score[current] + 1
                        if neighbor not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                            came_from[neighbor] = (current, move)
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + pow(pow(neighbor[0] - end[0], 2) + pow(neighbor[1] - end[1], 2), 0.5)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found
    
    def heuristic(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def astar_find_player(self, grid: player_board.PlayerBoard, start, end, enemy=False):
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        
        row, col = grid.get_dim_x(), grid.get_dim_y()
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}
        
        directions = {(-1, 0): Action.WEST, (1, 0): Action.EAST, (0, -1): Action.NORTH, (0, 1): Action.SOUTH, (-1, -1): Action.NORTHWEST, (-1, 1): Action.SOUTHWEST, (1, -1): Action.NORTHEAST, (1, 1): Action.SOUTHEAST}
        
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            board = grid
            if abs(current[0] - end[0]) <= 1 and abs(current[1] - end[1]) <= 1:
                # print(f"found at {current}")
                path = {}
                move_taken = None
                while current in came_from:
                    current, move_taken = came_from[current]
                    path[current] = move_taken

                return path


            # if current in visited:
            #     continue
            
            visited.add(current)

            prev_current = current
            moves_taken = []
            while prev_current in came_from:
                moves_taken.append(came_from[prev_current][1])
                prev_current = came_from[prev_current][0]
            moves_taken.reverse()
            for m in moves_taken:
                board, _ = board.forecast_move(m, sacrifice=1)

            for d in directions:
                neighbor = (current[0] + d[0], current[1] + d[1])
                                
                if 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
                    move = directions[d]
                    
                    if board.is_portal(neighbor[0], neighbor[1]):
                        # print(f"portal found at {neighbor}")
                        neighbor = tuple(int(i) for i in board.get_portal_dest(neighbor[0], neighbor[1]))
                    
                    if (board.is_valid_move(move, sacrifice=1, enemy=enemy)):
                        tentative_g_score = g_score[current] + 1
                        if neighbor not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                            came_from[neighbor] = (current, move)
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found