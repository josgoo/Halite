
import numpy as np  
import scipy.optimize
import sys
from copy import copy
from collections import defaultdict
import time

from kaggle_environments.envs.halite.helpers import *

PLUS_SHAPE = [(2,0),(1,1),(0,2),(1,-1),(0,-2),(-1,-1),(-2,0),(-1,1),(1,0),(0,1),(-1,0),(0,-1)] #Point distances for the plus shape
#The blur numbers were calculated through a Monte Carlo simulation of the probability a ship starting at 0,0 entered the square after 2 moves (completely randomly)
PLUS_SHAPE_BLUR = {(2,0): 0.04, (-2,0): 0.04, (1,1): 0.08, (0,2): 0.04, (1,-1): 0.08, (0,-2): 0.04,\
(-1,-1):0.08, (-1,1): 0.08, (1,0):0.24,(0,1):0.24,(-1,0):0.24,(0,-1):0.24,(0,0):1}
MOVES = [(1,0),(-1,0),(0,1),(0,-1),(0,0)] #Direct moves a ship can make
DIR_TO_ACTION = {(1,0): ShipAction.EAST, (-1,0): ShipAction.WEST, (0,1):ShipAction.NORTH, (0,-1): ShipAction.SOUTH, (0,0):None} #Direction to action dictionary


GENERAL_DOWNWEIGHT = 0.6 #Downweight for general enemy ships in dominance map
SPECIFIC_DOWNWEIGHT = 0.4 #Downweight for lighter enemy ships in dominance map (collisions)
BETA = 0.2 #Weighting to next turn miningLogic
TAU = 0.9 #Faith in our vectors

PREV = Point(-1,-1)
FUTURE_DROPOFFS = defaultdict(lambda: False)

DROPOFF_THRESHOLD = 1 #Necessary amortized value before returning
DISTANCE_THRESHOLD = 15 #How far away we look at points for mining

STORED_MOVES = 3 #How many previous enemy positions we store
STORED_ITER = 0 #counter that keeps track of past moves, wraps around (0-STORED_MOVES)
TEST_UGH = 0

enemy_squares = defaultdict(lambda: [None]*STORED_MOVES) #The stored enemy positions
enemy_vectors = {} #Stored resulting vector from enemy positions


PRINT_TIME = False


def createDominanceMap(board, ships):
    def updateGaussianBlur(ship, no_enemy_prob, thresh=float('inf')):
        for (x_dif, y_dif) in PLUS_SHAPE + [(0,0)]:
            square_point = (ship.position.x + x_dif, ship.position.y + y_dif)
            if ship.halite < thresh: #Always true for general case
                no_enemy_prob[square_point] *= (1- PLUS_SHAPE_BLUR[(x_dif,y_dif)]) #Probability no enemy enters the square
        return no_enemy_prob

    dominance_map = defaultdict(lambda: 1)
    for ship in board.ships.values():
        if ship.player_id != board.current_player_id:
            dominance_map = updateGaussianBlur(ship, dominance_map)
    for entry in dominance_map:
        dominance_map[entry] = (1 - dominance_map[entry]) * GENERAL_DOWNWEIGHT #The probability any enemy enters the square
    
    specific_dominance_map = defaultdict(lambda: {})
    for ship in ships:
        #Same as general downweight but only for ships that weigh more than itself
        #Going towards heavier ships would be the same, but I took it out because our local pathing doesn't account for those ships anymore
        specific_dominance_map[ship.id] = defaultdict(lambda: 1)
        for enemy_ship in board.ships.values():
            if enemy_ship.player_id != board.current_player_id:
                specific_dominance_map[ship.id] = updateGaussianBlur(enemy_ship, specific_dominance_map[ship.id], thresh=ship.halite)
        for entry in dominance_map:
            #1 - The probability at least 1 smaller enemy is in the area * downweight + general downweighting
            #Symbolizes value lost from dying (smaller ship) + value lost from them eating halite in the area (general)
            specific_dominance_map[ship.id][entry] = 1 - ( (1 - specific_dominance_map[ship.id][entry]) * SPECIFIC_DOWNWEIGHT + dominance_map[entry] )

    return specific_dominance_map #Each value runs somewhere between 0 - (GENERAL_DOWNEIGHT + SPECIFIC_DOWNWEIGHT)

def findAmortizedValueList(board, ship_point, dominance = None, printf=True):
    size = board.configuration.size
    targets = []
    #For every point on the board
    for point_x in range(size):
        for point_y in range(size):
            target = Point(point_x, point_y)
            H = board.cells[target].halite
            if H == 0:
                continue
            x_dist = abs(ship_point.x - point_x)
            y_dist = abs(ship_point.y - point_y)
            #length of traveling wrapping around the sides is the loop size - the forward facing path
            manhattan_distance = min(x_dist, size+1-x_dist) + min(y_dist, size+1-y_dist)
            #if manhattan_distance > DISTANCE_THRESHOLD:
            #    continue

            def neg_amortized_value(mining_time):
                if dominance:
                    return -( (1-.75**mining_time) * H ) / (manhattan_distance + mining_time) * dominance[(point_x, point_y)]
                return -( (1-.75**mining_time) * H ) / (manhattan_distance + mining_time)
            #Not rounding to nearest integer - may additionally account for uncertainty 
            #res=scipy.optimize.minimize_scalar(neg_amortized_value, bounds=(1,15),method='Bounded')
            top_val, best_mining_time = 0, -1
            for mining_time in range(1,15):
                a_val = neg_amortized_value(mining_time)
                if top_val > a_val:
                    top_val = a_val
                    best_mining_time = mining_time

            target_dict = {'point':target, 'value': -1*top_val} 
            targets.append(target_dict)
    #Doesn't currently break ties by putting shorter path first
    targets = sorted(targets, key=lambda x: x['value'], reverse=True) #sort squares by square_value
    return targets

def miningLogic(board, ships, dominance_map):
    global PREV, DISTANCE_THRESHOLD
    size = board.configuration.size
    
    targets = {}
    assigned = defaultdict(lambda: False)
    augmented = defaultdict(lambda: False)
    target_list = {}
    spot_loss = []
    #create list for every ship
    s = time.time()
    for ship in ships:
        target_list[ship] = findAmortizedValueList(board, ship.position, dominance_map[ship.id])
        ship_loss = (ship, target_list[ship][0]['value'] - target_list[ship][1]['value']) #tuple of (ship, ship_loss)
        spot_loss.append(ship_loss)
    assignment_order = sorted(spot_loss, key=lambda x: x[1], reverse=True) #sort spot_loss by loss
    e = time.time()
    print("mining part 1 time", e-s)
    #Assign every ship to its target in order
    b1,b2 = 0, 0
    for (ship, loss) in assignment_order:
        s1 = time.time()
        first = True
        #-- Get Top Value From target_list not assigned w/ augmentations--#
        while True:
            #if first:
            #    print('tl', target_list[ship][0], target_list[ship][1], target_list[ship][2])
            top = target_list[ship].pop(0)
            #if first:
            #    print("TOP", top)
            #    print('tv', augmented[top['point']], 'or', top['value'], 'nv' ,next_val)
            _next = target_list[ship][0]
            top_val = augmented[top['point']] if augmented[top['point']] else top['value']
            next_val = augmented[_next['point']] if augmented[_next['point']] else _next['value']
            if (not assigned[top['point']] and next_val <= top_val) or len(target_list[ship]) == 0:
                break
            first = False
        s2 = time.time()
        targets[ship] = {'point':top['point'], 'value':top_val, 'halite': board.cells[top['point']].halite}
        PREV = targets[ship]['point']
        assigned[top['point']] = True
        
        # ~~ Rerun of amortized analysis ~~ Potentially unnecessary
        next_target_list = findAmortizedValueList(board, top['point'], dominance_map[ship.id], printf=False)
        v1, v2 = next_target_list[0]['value'], next_target_list[1]['value']
        augmented[v1] = BETA * (v1 - v2)
        s3 = time.time()
        b1 += s2 - s1
        b2 += s3 - s2
    print("mining part 2 time", b1)
    print("mining part 3 time", b2)
    
    return (targets, assignment_order, augmented, assigned)

def factorCollisionsIntoActions(board, ship):
    weighting = defaultdict(lambda: 0)
    size = board.configuration.size
    for (x_dif, y_dif) in PLUS_SHAPE:
        square_point = Point( (ship.position.x + x_dif)%size , (ship.position.y + y_dif)%size)
        other_ship = board.cells[square_point].ship
        #TODO: ignores our ships, not sure how exactly to include it
        if other_ship and other_ship.player_id != board.current_player_id:
            enemy_vector = enemy_vectors[other_ship.id]
            prob_x = (enemy_vector[0] / STORED_MOVES) * TAU  + (1-TAU) * 0.2 #The proportion of times enemy moved in certain direction * confidence
            prob_y = (enemy_vector[1] / STORED_MOVES) * TAU + (1-TAU) * 0.2  # + (1-confidence) * random move in that direction
            prob_other = (1-TAU) * 0.2
            if other_ship.halite < ship.halite:
                for (x_move, y_move) in MOVES:
                    danger_point = (x_dif + x_move, y_dif + y_move)
                    if danger_point in MOVES:
                        positive_prob_x = x_move * prob_x #these are positive when the other ship's next move is
                        positive_prob_y = y_move * prob_y #in the same direction as it was previously going
                        if positive_prob_x > 0: 
                            weighting[danger_point] += positive_prob_x * -(500 + ship.halite)
                        elif positive_prob_y > 0:
                            weighting[danger_point] += positive_prob_y * -(500 + ship.halite)
                        else: #Expectation the ship stays still or moves in the opposite direction -> prob_other
                            weighting[danger_point] += prob_other * -(500 + ship.halite)
    return weighting

def findDesiredAction(board, ship, end, amortized_value, can_mine = True):
    start = ship.position
    if start == end:
        return [(0,0)] #None = Mine/Stay Still
    directions = {}
    total_dist = abs(end.x - start.x) + abs(end.y - start.y)
    vec_x = ( abs(end.x - start.x) / total_dist) 
    vec_y = ( abs(end.y - start.y) / total_dist)
    
    # (0,0) point is lower left hand corner of the board
    if end.x < start.x: #target is to the left of the ship
        vec_x *= -1
    if end.y < start.y: #target is below the ship
        vec_y *= -1
    collision_weightings = factorCollisionsIntoActions(board, ship)
    directions[(1,0)] = vec_x * amortized_value + collision_weightings[(1,0)]
    directions[(-1,0)] = -vec_x * amortized_value + collision_weightings[(-1,0)]
    directions[(0,1)] = vec_y * amortized_value + collision_weightings[(0,1)]
    directions[(0,-1)] = -vec_y * amortized_value + collision_weightings[(0,-1)]
    #staying still is divided by an additional 2 to account for the additional vec_x/vec_y weighting
    #If end.x-start.x equaled end.y-start.y then  it should be 1/2
    directions[(0,0)] = (board.cells[start].halite/4/2 if can_mine else 0) + collision_weightings[(0,0)]
    #print("DIRECTIONS", directions)
    return sorted(directions,key= lambda k: directions[k], reverse=True)

def assignMovesToShips(board, order, targets):
    open_space = defaultdict(lambda: True)
    for (ship, loss) in order:
        if decideDropoff(ship, targets):
            actions = findDesiredAction(board, ship, nearestDropoff(board, ship.position)['point'], 1, can_mine = False) #1 is a low ammortized value
        else:
            actions = findDesiredAction(board, ship, targets[ship]['point'], targets[ship]['value'])
        ship_point = ship.position
        while True:
            top_direction = actions.pop(0)
            new_point = Point(ship_point.x + top_direction[0], ship_point.y + top_direction[1])
            if open_space[new_point]:
                break
            if len(actions) == 0: #If this happens we are fucked, two friendly ships are colliding
                break

        open_space[new_point] = False
        ship.next_action = DIR_TO_ACTION[top_direction]

        global FUTURE_DROPOFFS
        if FUTURE_DROPOFFS[(ship.position.x, ship.position.y)]:
            ship.next_action = ShipAction.CONVERT
            print("CONVERTING:", (ship.position.x, ship.position.y))
            FUTURE_DROPOFFS[(ship.position.x, ship.position.y)] = False

       # print("SHIP ACTION", ship.next_action)
                
def decideDropoff(ship, targets):
    return targets[ship]['value'] < DROPOFF_THRESHOLD

def nearestDropoff(board, ship_point):
    size = board.configuration.size
    min_distance, min_pos = float('inf'), None
    for shipyard in board.current_player.shipyards:
        yard = shipyard.position
        x_dist = abs(ship_point.x - yard.x)
        y_dist = abs(ship_point.y - yard.y)
        #length of traveling wrapping around the sides is the loop size - the forward facing path
        distance = min(x_dist, size+1-x_dist) + min(y_dist, size+1-y_dist)
        if distance < min_distance:
            min_distance = distance
            min_pos = yard
    return {'dist':min_distance, 'point': min_pos}

def decideIfSpawnShip(board):
    
    #find current total halite on board
    H_t = 0
    s = len(board.ships)
    ms = len(board.current_player.ships)
    H,H1 = 0, 0
    for point_x in range(board.configuration.size):
        for point_y in range(board.configuration.size):
            target = Point(point_x, point_y)
            H_t += board.cells[target].halite
    H1_t = H_t     
    for t in range(400-board.step): #For t in the rest of the game
        x_t = H_t * s / 1600
        H_t = (H_t - x_t)*(1.02)
        H += x_t
        
        x1_t = H1_t * (s+1) / 1600
        H1_t = (H1_t - x1_t)*(1.02)
        H1 += x1_t
        
    ship_val = (ms+1) * H1 / (4 * (s+1)) - ms * H / (4 * s) if s > 0 else (ms+1) * H1 / (4 * (s+1))
    #print("NEW SHIP VAL:", ship_val)
    #print(" ship math", 'H1', H1 / (4 * (s+1)), 'H', H / (4 * s))
    if len(board.current_player.shipyards) > 0:
        board.current_player.shipyards[0].next_action = ShipyardAction.SPAWN
        #print("SPAWNED SHIP")
    return ship_val > 500

def SpawnShips2(board, augmented, assigned):
    for shipyard in board.current_player.shipyards:
        a_list = findAmortizedValueList(board, shipyard.position)
        while True:
            top = a_list.pop(0)
            _next = a_list[0]
            top_val = augmented[top['point']] if augmented[top['point']] else top['value']
            next_val = augmented[_next['point']] if augmented[_next['point']] else _next['value']
            if (not assigned[top['point']] and next_val <= top_val) or len(a_list) == 0:
                break
        if top_val * (400-board.step) * BETA/(1-BETA) > 500:
            shipyard.next_action = ShipyardAction.SPAWN
            print("SHIPYARD", shipyard.id, "SPAWNED with", top_val, 'a_val')


def decideIfCreateDropoff(board, ships, targets):
    size = board.configuration.size
    def dropoffSavings(square, ships):    
        #TODO: Implement code to create dropoff if None
        total_saved = 0
        print_total_saved = ""
        for ship in ships:
            ship_point = ship.position

            x_dist = abs(ship_point.x - square.x)
            y_dist = abs(ship_point.y - square.y)
            square_distance = min(x_dist, size+1-x_dist) + min(y_dist, size+1-y_dist)

            nearest_dropoff = nearestDropoff(board, ship_point)['dist']
            saved_amount = targets[ship]['value'] * max(nearest_dropoff - square_distance,0)
            total_saved += saved_amount
            x = max(nearest_dropoff - square_distance,0)
            print_total_saved += "( " + str(targets[ship]['value']) + " * " + str(x) + " ), "
        return (total_saved, print_total_saved)
    
    max_savings, best_dropoff = 0, None
    convert_savings, best_convert = 0, None
    max_print = ""

    for point_x in range(board.configuration.size):
        for point_y in range(board.configuration.size):
            square = Point(point_x, point_y)
            (savings, print_total_saved) = dropoffSavings(square, ships)

            ship_on_square = board.cells[square].ship
            if ship_on_square and ship_on_square.player_id == board.current_player_id and savings > convert_savings:
                convert_savings = savings
                best_convert = square
            if savings > max_savings:
                max_savings = savings
                best_dropoff = square
                max_print = print_total_saved
    print("TOTAL SAVED:", max_print)
    print(" savings", max_savings, 'max square', best_dropoff)
    if convert_savings == max_savings:
        best_dropoff = best_convert
    
    #print("DROPOFF STUFF (convert, max)", convert_savings, max_savings)
    global FUTURE_DROPOFFS
    
    if max_savings > 1000:
        FUTURE_DROPOFFS[(best_dropoff.x, best_dropoff.y)] = True
        print("FUTURE DROPOFF SET TO TRUE:", (best_dropoff.x, best_dropoff.y))
    return

def updateEnemyVectors(board):
    STORED_ITER = board.step % STORED_MOVES
    for ship in board.ships.values():
        if ship.player_id != board.current_player_id:
            vector = Point(0,0)
            for i in range(STORED_MOVES):
                s = enemy_squares[ship.id][ (STORED_ITER + i) % STORED_MOVES ]
                e = enemy_squares[ship.id][ (STORED_ITER + i + 1) % STORED_MOVES ]
                if i < STORED_MOVES - 1 and s and e: #if the ship hasn't been alive, s/e will be None
                    move = s - e
                    vector += move
                elif i == STORED_MOVES - 1 and s: #the last move is from the last recorded position to the current one
                    move = ship.position - s
                    vector += move #add all incremental moves to create vector
            enemy_squares[ship.id][STORED_ITER] = ship.position
            enemy_vectors[ship.id] = vector

def agent(obs, config):

    start = time.time()
    size = config.size
    board = Board(obs, config)
    my = board.current_player
    ships = my.ships
    print("\nSHIP POSITIONS: ")
    for ship in ships:
        print("  ", ship.position)
    updateEnemyVectors(board)
    uev = time.time()
    #print("Update Enemy Vectors:", round(uev-start, 2), 'total:', round(uev-start,2))

    dominance_map = createDominanceMap(board, ships)
    dom = time.time()
    #print("Dominance Map:", round(dom-uev, 2), 'total:', round(dom-start,2))


    (targets, assignment_order, augmented, assigned) = miningLogic(board, ships, dominance_map)
    mining = time.time()
    print("Mining Logic:", round(mining-dom, 2), 'total:', round(mining-start,2))
    #for ship in targets:
    #    print("  ",ship.position, '->' ,targets[ship])

    decideIfCreateDropoff(board, ships, targets)
    df = time.time()
    #print("Create dropoff:", round(df-mining, 2), 'total:', round(df-start,2))

    assignMovesToShips(board, assignment_order, targets)
    assign = time.time()
   # print("Assign Moves:", round(assign-df, 2), 'total:', round(assign-start,2))
    
    #decideIfSpawnShip(board)
    SpawnShips2(board, augmented, assigned)
    end =time.time()
    print("Spawn Ships:", round(end-assign, 2), 'total:', round(end-start,2))
    print("\n\n")
  #  print("TIME TOOK", end-start)
    return my.next_actions
