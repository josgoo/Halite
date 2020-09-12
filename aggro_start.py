import numpy as np
import scipy.optimize
from collections import defaultdict
import time
import json
from kaggle_environments.envs.halite.helpers import *

# constants
PLUS_SHAPE = [(2,0),(1,1),(0,2),(1,-1),(0,-2),(-1,-1),(-2,0),(-1,1),(1,0),(0,1),(-1,0),(0,-1)] #Point distances for the plus shape
#The blur numbers were calculated through a Monte Carlo simulation of the probability a ship starting at 0,0 entered the square after 2 moves (completely randomly)
PLUS_SHAPE_BLUR = {(2,0): 0.04, (-2,0): 0.04, (1,1): 0.08, (0,2): 0.04, (1,-1): 0.08, (0,-2): 0.04,\
(-1,-1):0.08, (-1,1): 0.08, (1,0):0.24,(0,1):0.24,(-1,0):0.24,(0,-1):0.24,(0,0):1}
MOVES = [ (1,0), (-1,0), (0,1), (0,-1), (0,0) ] #Direct moves a ship can make
DIR_TO_ACTION = {(1,0): ShipAction.EAST, (-1,0): ShipAction.WEST, (0,1):ShipAction.NORTH, (0,-1): ShipAction.SOUTH, (0,0):None} #Direction to action dictionary

# hyperparams
GENERAL_DOWNWEIGHT = 0.25 #Downweight for general enemy ships in dominance map
SPECIFIC_DOWNWEIGHT = 0.75 #Downweight for lighter enemy ships in dominance map (collisions)
BETA = 0.3 #Weighting to next turn miningLogic
TAU = 0.25 #Faith in our vectors
GAMMA_CHANGE = 120 #Turn where we use the other gamma
GAMMA1_TUNE, GAMMA2_TUNE = -19,-0.56
GAMMA1_CONST, GAMMA2_CONST = 92.5,4.36 #Tuning parameter for gamma
GOLDEN_RATIO = 2 #Ratio of how much our amortized value overvalues how much halite we will actually recieve
SHIP_SPAWN_LIMIT = 370
DISTANCE_THRESHOLD = 6 #How far away we look at points to find dropoffs
DROPOFF_VISION = 4 #Distance a dropoff can look
DROPOFF_LIMIT = 3 #How many ships we pretend to spawn to pick best dropoff location
MAX_SHIP_TO_YARD_RATIO = 11#15#9
DISTANCE_BETWEEN_DROPOFFS = 7
SHIPS_REQUIRED_IN_DROPOFF_DISTANCE = 2
STORED_MOVES = 3 #How many previous enemy positions we store
STORED_ITER = 0 #counter that keeps track of past moves, wraps around (0-STORED_MOVES)
MAX_STEPS = 400
MAX_SHIPS = 40 #Max ships we build to accomodate for the turn timer
STORED_HALITE_VALUE = 0.01
STORED_HALITE_INCR = 0.04
MAX_STORED_HALITE_VALUE = 0.75
PROTECT_THRESHOLD = 250
ESHIP_VAL = 200#125
MAX_ATTACKERS_TO_SHIP = 6
EXPECTED_CAPTURE_TIME = [float('inf'),64,16,4,2,1]
OPPONENT_ATTACK_MIN_SHIPS = 3 #Maybe change to somehow only count ships in starting quadrant
WITHIN_BOUNDARY_MULTIPLIER = 4

# global variables
FUTURE_DROPOFF = None
BEST_NEW_DROPOFF = None
RETURNING = defaultdict(lambda: False)
enemy_squares = defaultdict(lambda: [None]*STORED_MOVES) #The stored enemy positions
enemy_vectors = {} #Stored resulting vector from enemy positions
total_mined = defaultdict(lambda: (0,0))
N_ATTACKING = defaultdict(lambda: 0)
PREV_ATTACK_TARGETS = {}
SQUARE_AVALS = defaultdict(lambda: 0)
OPPONENT_TO_TARGET = None
shipyard_occupied = defaultdict(lambda: defaultdict(lambda: None)) #ship scheduled to be on a shipyard on a certain turn
NEAREST_DROPOFF = {}
ATTACKING_SHIPS = defaultdict(lambda: False)
DANGER = defaultdict(lambda: [0,0,0])
CENTER, CENTER_VAL = None, float('inf')
HAS_DECIMATED = -1
BORDERS = {}

log = []
logging_mode = True
print_log = True

def shipAttackValue(board, ship_pos, attack_point_vals):
    ship = board.cells[ship_pos].ship
    specific_dominance = defaultdict(lambda: 1)

    this_attack_vals = defaultdict(lambda: {'v': [0]*MAX_ATTACKERS_TO_SHIP,'target':None})
    for e_ship in attack_point_vals:
        for square in attack_point_vals[e_ship]:
            dist = manhattan_distance(ship_pos, Point(square[0],square[1]))
            if dist == 0 and board.cells[ ship_pos ].halite > 0:
                continue
            for i in range(MAX_ATTACKERS_TO_SHIP):
                attack_ship_square_val = attack_point_vals[e_ship][square] / (dist + EXPECTED_CAPTURE_TIME[i]) / (i+1) * specific_dominance[(square[0], square[1])]
                #only assign square to target ship if there were no other ships chasing it
                if i == 0 and attack_ship_square_val > this_attack_vals[square]['v'][0]:
                    this_attack_vals[square]['target'] = e_ship
                if this_attack_vals[square]['target'] == e_ship:
                    this_attack_vals[square]['v'][i] = attack_ship_square_val

    return this_attack_vals
def bestAttackTarget(ship_attack_vals, targeted):
    global N_ATTACKING
    max_val, max_square = 0, None
    for square in ship_attack_vals:
        n = N_ATTACKING[ ship_attack_vals[square]['target'] ]
        if not targeted[square] and n < MAX_ATTACKERS_TO_SHIP and max_val < ship_attack_vals[square]['v'][n]:
            max_val =ship_attack_vals[square]['v'][n]
            max_square = square
    return (max_square, max_val)
def moveToPlus(move_prob, plus):
    if plus == (2,0):
        return move_prob[(1,0)] * move_prob[(1,0)]
    elif plus == (0,2):
        return move_prob[(0,1)] * move_prob[(0,1)]
    elif plus == (-2,0):
        return move_prob[(-1,0)] * move_prob[(-1,0)]
    elif plus == (0,-2):
        return move_prob[(0,-1)] * move_prob[(0,-1)]
    elif plus == (1,1):
        return 2 * move_prob[(0,1)] * move_prob[(1,0)]
    elif plus == (-1,1):
        return 2 * move_prob[(0,1)] * move_prob[(-1,0)]
    elif plus == (1,-1):
        return 2 * move_prob[(0,-1)] * move_prob[(1,0)]
    elif plus == (-1,-1):
        return 2 * move_prob[(0,-1)] * move_prob[(-1,0)]
    elif plus == (0,1):
        return move_prob[(0,1)] + move_prob[(0,0)] * move_prob[(0,1)]
    elif plus == (1,0):
        return move_prob[(1,0)] + move_prob[(0,0)] * move_prob[(1,0)]
    elif plus == (0,-1):
        return move_prob[(0,-1)] + move_prob[(0,0)] * move_prob[(0,-1)]
    elif plus == (-1,0):
        return move_prob[(-1,0)] + move_prob[(0,0)] * move_prob[(-1,0)]
    elif plus == (0,0):
        return 1

def getAdjacentOpponents(board):
    me_id = board.current_player.id
    if me_id == 0 or me_id == 3:
        return [1,2]
    else:
        return [0,3]


def isDecimated(board):
    if OPPONENT_TO_TARGET == None:
        return False
    opponent = board.players[OPPONENT_TO_TARGET]
    MIN_SHIPS_TO_DECIMATE, MIN_DIF_TO_DECIMATE = 20, 12
    decimated = len(board.current_player.ships) >= MIN_SHIPS_TO_DECIMATE and len(board.current_player.ships) - len(opponent.ships) >= MIN_DIF_TO_DECIMATE
    if decimated:
        global HAS_DECIMATED
        HAS_DECIMATED = OPPONENT_TO_TARGET
    return decimated

def chooseOpponentToDecimateViaRatio(board):
    global HAS_DECIMATED, OPPONENT_TO_TARGET
    if HAS_DECIMATED >= 0 or isDecimated(board):
        OPPONENT_TO_TARGET = None
        return
    max_ratio, max_opponent = 0, None
    neighbors = getAdjacentOpponents(board)
    for opponent in board.opponents:
        if opponent.id in neighbors:
            miners = sum([1 if e_ship.halite > 0 else 0 for e_ship in opponent.ships])
            total_cargo = sum([e_ship.halite for e_ship in opponent.ships])
            ratio = total_cargo / len(opponent.ships) if len(opponent.ships) > 0 else 0
            if ratio > max_ratio and miners > OPPONENT_ATTACK_MIN_SHIPS + 2:
                max_ratio = ratio
                max_opponent = opponent.id

    OPPONENT_TO_TARGET = max_opponent

def getBorders(board):
        if HAS_DECIMATED >= 0:
            me = board.current_player.id
            if me + HAS_DECIMATED == 1:
                return ((0,20), (20,10))
            elif me + HAS_DECIMATED == 2:
                return ((0,20), (10,0))
            elif me + HAS_DECIMATED == 4:
                return ((10,20), (20,0))
            elif me + HAS_DECIMATED == 5:
                return ((0,10), (20,0))
            print("ERROR" + str(HAS_DECIMATED + me))
            return "ERROR" + str(HAS_DECIMATED + me)
        if CENTER != None:
            size = board.configuration.size
            tlx = (CENTER.x - 5) % size
            tly = (CENTER.y + 5) % size
            brx = (CENTER.x + 5) % size
            bry = (CENTER.y - 5) % size
            return ((tlx,tly), (brx,bry))

        global BORDERS
        if len(BORDERS) > 0:
            return BORDERS

        within = defaultdict(lambda: False)
        size = board.configuration.size
        for x in range(size):
            for y in range(size):
                square = Point(x,y)
                friendly_dist = nearestDropoff(board, square)['orig_dist']
                enemy_dist = nearestEDropoff(board, square)
                if friendly_dist < enemy_dist:
                    within[square] = True
        BORDERS = within
        return within

def withinBorders(board, point):
    borders = getBorders(board)
    if isinstance(borders, dict):
        return borders[point]
    #if borders is not a dictionary, it must be the half of the map
    ((tlx,tly), (brx,bry)) = borders
    if tlx < brx and (point.x < tlx or point.x > brx):
        return False
    if brx < tlx and (brx < point.x < tlx):
        return False
    if tly > bry and (point.y > tly or point.y < bry):
        return False
    if tly < bry and (tly < point.y < bry):
        return False
    return True
def distanceDiscount(board, point):
    return 1
    nearest_dropoff = nearestDropoff(board, point)
    discount = float( np.power(.97, nearest_dropoff['dist']) )
    return discount

def isPastAttackingTime(board):
    return board.step >= 80 or len(board.current_player.ships) >= 16

def attackLogic(board, attacking_ships):
    global logging_mode
    attack_log = {}
    if logging_mode:
        attack_point_vals_log = {}

    attack_point_vals = {}
    size = board.configuration.size
    
    global EXPECTED_CAPTURE_TIME, PREV_ATTACK_TARGETS
    if isPastAttackingTime(board): #begin to attack
        EXPECTED_CAPTURE_TIME[0] = 128

    if isPastAttackingTime(board) and OPPONENT_TO_TARGET == None: #choose a potential target if none is selected
        chooseOpponentToDecimateViaRatio(board)
    if OPPONENT_TO_TARGET != None: #re-evaluate target if the chosen target isn't advantageous enough
        miners = sum([1 if e_ship.halite > 0 else 0 for e_ship in board.players[OPPONENT_TO_TARGET].ships])
        chooseOpponentToDecimateViaRatio(board)

    #assumes all attack ships have 0 halite
    #populate value of probability of capturing each ship at every point
    target_opponents = [board.players[OPPONENT_TO_TARGET]] if OPPONENT_TO_TARGET != None else board.opponents
    for opponent in target_opponents:
        for e_ship in opponent.ships:
            if e_ship.halite > 0:
                attack_point_vals[e_ship.id] = defaultdict(lambda: 0)
                if logging_mode:
                    attack_point_vals_log[e_ship.id] = defaultdict(lambda: 0)
                
                e_ship_pos = e_ship.position
                (prob_x, prob_y, prob_other) = expectedShipAction(board, e_ship)
                move_prob = {}
                for (x_move, y_move) in MOVES:
                    possible_point = Point( (e_ship_pos.x + x_move)%size , (e_ship_pos.y + y_move)%size )
                    positive_prob_x = x_move * prob_x #these are positive when the other ship's next move is
                    positive_prob_y = y_move * prob_y #in the same direction as it was previously going
                    if positive_prob_x > 0:
                        prob_actualized = positive_prob_x
                    elif positive_prob_y > 0:
                        prob_actualized = positive_prob_y
                    else: #Expectation the ship stays still or moves in the opposite direction -> prob_other
                        prob_actualized = prob_other
                    move_prob[(x_move, y_move)] = prob_actualized
                for (x_move, y_move) in MOVES:
                    possible_point = Point( (e_ship_pos.x + x_move)%size , (e_ship_pos.y + y_move)%size )
                    prob_actualized = move_prob[(x_move, y_move)]
                    for (rel_x, rel_y) in PLUS_SHAPE_BLUR:
                        attack_point = ( (possible_point.x + rel_x)%size , (possible_point.y + rel_y)%size )
                        capture_prob = moveToPlus( move_prob, (rel_x, rel_y) )
                        ap = Point(attack_point[0], attack_point[1])
                        within_multiplier = WITHIN_BOUNDARY_MULTIPLIER * distanceDiscount(board, ap) if isPastAttackingTime(board) and OPPONENT_TO_TARGET == None and withinBorders(board, ap) else 1
                        attack_point_vals[e_ship.id][attack_point] += capture_prob * prob_actualized * (ESHIP_VAL + e_ship.halite) * within_multiplier

                        if logging_mode:
                            attack_point_vals_log[e_ship.id][str(attack_point)] += capture_prob * prob_actualized * (ESHIP_VAL + e_ship.halite) * within_multiplier

    if logging_mode:
        attack_log['e_ship, point'] = attack_point_vals_log
    #populate value of each ship moving to and capturing any ship for every point
    attack_vals = {}
    if logging_mode:
        attack_vals_log = {}

    for ship in attacking_ships:
        if RETURNING[ship.id] or ship.halite > 0:
            continue
        attack_vals[ship.id] = shipAttackValue(board, ship.position, attack_point_vals)

        if logging_mode:
            attack_vals_log[ship.id] = {}
            for square in attack_vals[ship.id]:
                attack_vals_log[ship.id][str(square)] = attack_vals[ship.id][square]
    
    if logging_mode:
        attack_log['f_ship, point'] = attack_vals_log
        log_targeted = defaultdict(lambda: False)
    #assign targets based on the list
    attack_targets = {}
    targeted = defaultdict(lambda: False)
    attack_order = []
    #assign targets to every attacking ship unless we have already assigned all attackers
    while len(attack_targets) < len(attacking_ships) and len(attack_targets) < MAX_ATTACKERS_TO_SHIP * len(attack_point_vals):
        max_square, max_ship, max_val = None,None,0
        for ship in attacking_ships:
            if RETURNING[ship.id] or ship.halite > 0:
                continue
            if ship.id not in attack_targets:
                (best_square, best_val) = bestAttackTarget(attack_vals[ship.id], targeted)
                if best_val > max_val:
                    max_square = best_square
                    max_ship = ship
                    max_val = best_val
        #If we assigned all available ships
        if max_ship == None:
            break
        target_ship = attack_vals[max_ship.id][ max_square ]['target']
        attack_targets[max_ship.id] = {'point': max_square, 'value': max_val, 'target': target_ship}
        #Adjust number of attacking each ship
        N_ATTACKING[target_ship] += 1
        if max_ship.id in PREV_ATTACK_TARGETS and PREV_ATTACK_TARGETS[max_ship.id]['target']:
            N_ATTACKING[ PREV_ATTACK_TARGETS[max_ship.id]['target'] ] -= 1
        targeted[max_square] = True
        attack_order.append(max_ship.id)

        if logging_mode:
            log_targeted[str(max_square)] = True
    
    if logging_mode:
        attack_log['targets'] = attack_targets
        attack_log['targeted'] = log_targeted
        attack_log['n_attacking'] = N_ATTACKING

    no_targets_assigned = []
    global ATTACKING_SHIPS, FUTURE_DROPOFF
    for ship in attacking_ships:
        if ship.id not in attack_targets:
            #Force return ships with more than 0 halite
            if RETURNING[ship.id] or ship.halite > 0:
                nearest_dropoff = nearestDropoff(board, ship.position, h=ship.halite)
                #if a shipyard exists
                if nearest_dropoff['point'] != None:
                    attack_targets[ship.id] = {'point':nearest_dropoff['point'], 'value': 1, 'target': None}
                    RETURNING[ship.id] = True
                    #returning ships have priority in attack order
                    attack_order.insert(0, ship.id)
                    nearest_shipyard = board.cells[ nearest_dropoff['point'] ].shipyard
                    if nearest_shipyard:
                        shipyard_occupied[ nearest_shipyard.id  ][board.step + nearest_dropoff['dist']] = ship.id
                    if BEST_NEW_DROPOFF and FUTURE_DROPOFF != None and nearest_dropoff['point'] == BEST_NEW_DROPOFF:
                        FUTURE_DROPOFF = BEST_NEW_DROPOFF
                else: #make it a mining ship
                    no_targets_assigned.append(ship)
                    ATTACKING_SHIPS[ship.id] = False
            #If a ship wasn't assigned and has 0 halite, make it a mining ship
            else:
                no_targets_assigned.append(ship)
                ATTACKING_SHIPS[ship.id] = False
    PREV_ATTACK_TARGETS = attack_targets
    return (attack_targets, targeted, attack_point_vals, attack_order, no_targets_assigned, attack_log)
def createDominanceMap(board, ships):
    size = board.configuration.size
    def updateGaussianBlur(ship_pos, ship_halite , no_enemy_prob, thresh=float('inf'), attacking=False):
        for (x_dif, y_dif) in PLUS_SHAPE + [(0,0)]:
            square_point = ( int( (ship_pos.x + x_dif)%size ), int( (ship_pos.y + y_dif)%size ) )
            if thresh != float('inf') and ship_halite <= thresh and (x_dif, y_dif) in MOVES and not attacking:
                no_enemy_prob[square_point] = 0 #if there is a lighter ship in the specific case within 1 move (count it as just as dangerous)
            elif ship_halite < thresh: #Always true for general case
                no_enemy_prob[square_point] *= (1- PLUS_SHAPE_BLUR[(x_dif,y_dif)]) #Probability no enemy enters the square
        return no_enemy_prob
    dominance_map = defaultdict(lambda: 1)
    for ship in board.ships.values():
        if ship.player_id != board.current_player_id:
            dominance_map = updateGaussianBlur(ship.position, ship.halite, dominance_map)
    for shipyard in board.shipyards.values():
        if shipyard.player_id != board.current_player_id:
             dominance_map = updateGaussianBlur(shipyard.position, float('inf'), dominance_map, thresh=0)
    for entry in dominance_map:
        dominance_map[entry] = float( 1 - ( (1 - dominance_map[entry]) * GENERAL_DOWNWEIGHT) ) #The probability any enemy enters the square

    specific_dominance_map = {}
    updated_zero_dominance = False
    for ship in ships:
        #Same as general downweight but only for ships that weigh more than itself
        #Going towards heavier ships would be the same, but I took it out because our local pathing doesn't account for those ships anymore
        specific_dominance_map[ship.id] = defaultdict(lambda: 1)
        is_attacking =  ship.halite == 0 and ATTACKING_SHIPS[ship.id]
        for enemy_ship in board.ships.values():
            if enemy_ship.player_id != board.current_player_id:
                specific_dominance_map[ship.id] = updateGaussianBlur(enemy_ship.position,enemy_ship.halite, specific_dominance_map[ship.id], thresh=ship.halite, attacking =  is_attacking)
        for shipyard in board.shipyards.values():
            if shipyard.player_id != board.current_player_id:
                 specific_dominance_map[ship.id] = updateGaussianBlur(shipyard.position, float('inf'), specific_dominance_map[ship.id], thresh=ship.halite, attacking =  is_attacking)
        for entry in dominance_map:
            #1 - The probability at least 1 smaller enemy is in the area * downweight + general downweighting
            #Symbolizes value lost from dying (smaller ship) + value lost from them eating halite in the area (general)
            if ship.halite == 0 and ATTACKING_SHIPS[ship.id]:
                specific_dominance_map[ship.id][entry] = float( 1 - ( (1 - specific_dominance_map[ship.id][entry]) * SPECIFIC_DOWNWEIGHT) )
            else:
                specific_dominance_map[ship.id][entry] = float( dominance_map[entry] - ( (1 - specific_dominance_map[ship.id][entry]) * SPECIFIC_DOWNWEIGHT ) )
    return (dominance_map, specific_dominance_map) #Each value runs somewhere between 0 - (GENERAL_DOWNEIGHT + SPECIFIC_DOWNWEIGHT)
def findAmortizedValueList(board, ship_point, dominance = None, max_dist=21, store=True, discount_distance = True):
    size = board.configuration.size
    targets = []
    global SQUARE_AVALS
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
            manhattan_distance = min(x_dist, size-x_dist) + min(y_dist, size-y_dist)
            if manhattan_distance > max_dist:
                continue

            def neg_amortized_value(mining_time):
                val = -( (1-.75**mining_time) * H ) / (manhattan_distance + mining_time)
                if dominance:
                    val *= dominance[(point_x, point_y)]
                if isPastAttackingTime(board) and OPPONENT_TO_TARGET == None and withinBorders(board, target):
                    val *= WITHIN_BOUNDARY_MULTIPLIER
                if board.step < 30:
                    starting_locations = [Point(5,15), Point(15,15), Point(5,5), Point(15,5)]
                    first_dropoff_dist = squareDistance(starting_locations[board.current_player.id], target) if len(board.current_player.shipyards) > 0 else 1
                    if first_dropoff_dist <= 10:
                        val *= (first_dropoff_dist)
                return val
            #Not rounding to nearest integer - may additionally account for uncertainty
            top_val, best_mining_time = 0, -1
            mining_limit = min(MAX_STEPS - board.step - 2 * manhattan_distance, 15)#approximation for running nearestDropoff
            for mining_time in range(1, mining_limit):
                a_val = neg_amortized_value(mining_time)
                if top_val > a_val:
                    top_val = a_val
                    best_mining_time = mining_time
            if store:
                SQUARE_AVALS[(point_x, point_y)] = max( SQUARE_AVALS[(point_x, point_y)],  -1*top_val/GOLDEN_RATIO)
            final_val = -1*top_val/GOLDEN_RATIO
            target_dict = {'point':target, 'value': final_val, 'mining_time': manhattan_distance + best_mining_time, 'mined': (1-.75**best_mining_time) * H/GOLDEN_RATIO}
            targets.append(target_dict)
    #Doesn't currently break ties by putting shorter path first
    targets = sorted(targets, key=lambda x: x['value'], reverse=True) #sort squares by square_value
    return targets
def miningLogic(board, ships, dominance_map, assigned_attacks):
    global DISTANCE_THRESHOLD
    size = board.configuration.size
    assigned = assigned_attacks.copy()
    target_list = {}
    spot_loss = []
    #create list for every ship
    s = time.time()
    for ship in ships:
        target_list[ship.id] = findAmortizedValueList(board, ship.position, dominance_map[ship.id])
        ship_loss = (ship.id, target_list[ship.id][0]['value'] - target_list[ship.id][1]['value']) #tuple of (ship, ship_loss)
        spot_loss.append(ship_loss)
    assignment_order = sorted(spot_loss, key=lambda x: x[1], reverse=True) #sort spot_loss by loss
    shipyard_ships, non_shipyard_ships = [], []
    for (ship_id, loss) in assignment_order:
        ship = board.ships[ship_id]
        on_shipyard = False
        for shipyard in board.current_player.shipyards:
            if ship.position == shipyard.position:
                on_shipyard = True
                break
        if on_shipyard:
            shipyard_ships.append(ship_id)
        else:
            non_shipyard_ships.append(ship_id)
    #assign targets to ships on shipyards last
    revised_order = non_shipyard_ships + shipyard_ships
    targets = {}
    augmented = defaultdict(lambda: False)
    for ship_id in revised_order:
        ship = board.ships[ship_id]
        s1 = time.time()
        #-- Get Top Value From target_list not assigned w/ augmentations--#
        topped = []
        idx = 0
        while True:
            top = target_list[ship_id][idx]
            _next = target_list[ship_id][idx+1]
            topped.append(top)
            top_val = augmented[top['point']] if augmented[top['point']] else top['value']
            next_val = _next['value'] - augmented[_next['point']] if augmented[_next['point']] else _next['value']
            idx += 1
            if (not assigned[top['point']] and next_val <= top_val) or idx >= len(target_list[ship_id]) - 1:
                break
        s2 = time.time()
        targets[ship_id] = {'point':top['point'], 'value':top_val, 'halite': board.cells[top['point']].halite, 'mining_time':top['mining_time'], 'mined': top['mined'], 'next_val': 0}
        assigned[top['point']] = True

        if len(board.current_player.ships) < 30:
            # ~~ Rerun of amortized analysis ~~ Potentially unnecessary
            next_target_list = findAmortizedValueList(board, top['point'], dominance_map[ship.id])
            #obv we will stay at the point (its the best point from a distance, must be best if we are on it), so take next
            p1, v1, v2 = next_target_list[1]['point'], next_target_list[1]['value'], next_target_list[2]['value']
            targets[ship_id]['next_val'] = v1
            augmented[p1] = BETA * (v1 - v2)
            s3 = time.time()

    def isMining(ship):
        try:
            return targets[ship.id]['point'] == ship.position
        except KeyError:
            return False
    #swap targets if ships are blocking all routes to a destination
    for i in range(len(revised_order)-1,-1,-1):
        ship = board.ships[ revised_order[i] ]
        if isMining(ship):
            continue
        (vec_x,vec_y) = findVectorComponents(board, ship.position, targets[ship.id]['point'])
        vec_x, vec_y = int(np.sign(vec_x)), int(np.sign(vec_y))
        x_blocked = vec_x == 0
        point_x = Point((ship.position.x + vec_x)%size, ship.position.y)
        ship_x = board.cells[point_x].ship
        if ship_x and ship_x.player_id == board.current_player_id and isMining(ship_x):
            x_blocked = True

        y_blocked = vec_y == 0
        point_y = Point(ship.position.x, (ship.position.y + vec_y)%size)
        ship_y = board.cells[point_y].ship
        if ship_y and ship_y.player_id == board.current_player_id and isMining(ship_y):
            y_blocked = True

        if x_blocked and y_blocked:
            ship_to_move = ship_x if abs(vec_x) >= abs(vec_y) else ship_y
            temp = targets[ship_to_move.id]
            targets[ship_to_move.id] = targets[ship.id]
            targets[ship.id] = temp
            targets[ship.id]['exchanged'] = True
    return (targets, target_list, revised_order, augmented, assigned)


def assignTaskToShips(board, targets, attack_point_vals, general_dominance_map, augmented, assigned):
    global ATTACKING_SHIPS
    CONVERT_FACTOR = 14 if board.step < 300 else 1
    assign_log = {}
    for ship in board.current_player.ships:
        if not RETURNING[ship.id] and not ATTACKING_SHIPS[ship.id]:
            #TODO: Cover case where multiple friendly ships would want to switch to attacking at once but not individually
            this_attack_vals = shipAttackValue(board, ship.position, attack_point_vals)
            (attack_target, attack_val) = bestAttackTarget(this_attack_vals, assigned)
            if targets['mine'][ship.id]['value'] < attack_val * CONVERT_FACTOR/2:
                ATTACKING_SHIPS[ship.id] = True
            assign_log[ship.id] = {'started': 'mine', 'ended_same': not ATTACKING_SHIPS[ship.id], 'mine_val': targets['mine'][ship.id]['value'],\
                         'attack_val': attack_val}

        elif not RETURNING[ship.id] and ATTACKING_SHIPS[ship.id]:
            (mining_target, mining_val) = newMiningShipValue(board, ship.position, general_dominance_map, augmented, assigned)
            if targets['attack'][ship.id]['value'] * CONVERT_FACTOR < mining_val:
                ATTACKING_SHIPS[ship.id] = False
                assigned[mining_target] = True
            assign_log[ship.id] = {'started': 'attack', 'ended_same': ATTACKING_SHIPS[ship.id], 'mine_val': mining_val, 'attack_val': targets['attack'][ship.id]['value']}
    return assign_log

def expectedShipAction(board, other_ship):
    enemy_vector = enemy_vectors[other_ship.id]
    prob_other = (1-TAU) * 0.2
    prob_x = (enemy_vector[0] / STORED_MOVES) * TAU#The proportion of times enemy moved in certain direction * confidence
    prob_x = prob_x + prob_other if prob_x >= 0 else prob_x - prob_other
    prob_y = (enemy_vector[1] / STORED_MOVES) * TAU  # + (1-confidence) * random move in that direction
    prob_y = prob_y + prob_other if prob_y >= 0 else prob_y - prob_other
    return(prob_x, prob_y, prob_other)
def factorCollisionsIntoActions(board, ship, capture_cost):
    global DANGER
    weighting = defaultdict(lambda: 0)
    DANGER[ship.id][board.step% len(DANGER[ship.id])] = 0
    size = board.configuration.size
    COLLISION_COEF = 5 #Giving the chance they move into a territory to collide with us as more likely
    for (x_dif, y_dif) in PLUS_SHAPE:
        square_point = Point( (ship.position.x + x_dif)%size , (ship.position.y + y_dif)%size)
        other_ship = board.cells[square_point].ship
        #TODO: ignores our ships, not sure how exactly to include it
        if other_ship and other_ship.player_id != board.current_player_id:
            (prob_x, prob_y, prob_other) = expectedShipAction(board, other_ship)
            if other_ship.halite <= ship.halite:
                for (x_move, y_move) in MOVES:
                    danger_point = ( x_dif + x_move , y_dif + y_move )
                    if danger_point in MOVES:
                        if (x_dif, y_dif) in MOVES:
                            DANGER[ship.id][board.step%2] += 2
                        else:
                            DANGER[ship.id][board.step%2] += 1
                        positive_prob_x = x_move * prob_x #these are positive when the other ship's next move is
                        positive_prob_y = y_move * prob_y #in the same direction as it was previously going
                        if weighting[other_ship.id] == 0:
                            weighting[other_ship.id] = [( (int(x_move), int(y_move)), (int(x_dif),int(y_dif)) , prob_x, prob_y)]
                        else:
                            weighting[other_ship.id].append( ( (int(x_move), int(y_move)), (int(x_dif),int(y_dif)) , prob_x, prob_y) )
                        if positive_prob_x > 0:
                            weighting[danger_point] += COLLISION_COEF * positive_prob_x * -capture_cost
                        elif positive_prob_y > 0:
                            weighting[danger_point] += COLLISION_COEF * positive_prob_y * -capture_cost
                        else: #Expectation the ship stays still or moves in the opposite direction -> prob_other
                            if other_ship.halite == 0 and board.cells[ other_ship.position ].halite > 0 and (x_move,y_move) == (0,0):
                                weighting[danger_point] += prob_other * -capture_cost #lighter penalty because they will most likely not stay still
                            else:
                                weighting[danger_point] += COLLISION_COEF * prob_other * -capture_cost
        other_shipyard = board.cells[square_point].shipyard
        if other_shipyard and other_shipyard.player_id != board.current_player_id:
            weighting[(x_dif, y_dif)] += -capture_cost #TODO: maybe add value of destroying shipyard
    return weighting
def findVectorComponents(board, start, end):
    if start == end:
        return (0,0)
    size = board.configuration.size
    x_mult, y_mult = 1, 1
    x_dist = abs(start.x - end.x)
    if x_dist > size-x_dist:
        x_mult = -1
        x_dist = size-x_dist
    y_dist = abs(start.y - end.y)
    if y_dist > size-y_dist:
        y_mult = -1
        y_dist = size-y_dist
    #length of traveling wrapping around the sides is the loop size - the forward facing path
    total_dist = x_dist + y_dist
    vec_x = (end.x - start.x) / total_dist * x_mult
    vec_y = (end.y - start.y) / total_dist * y_mult
    return (vec_x, vec_y)
def findDesiredAction(board, ship, end, amortized_value, can_mine = True):
    start = ship.position
    directions = {}
    (vec_x, vec_y) = findVectorComponents(board,start, end)
    collision_cost = 500 + ship.halite + amortized_value - gamma_pdf(board.step)
    collision_weightings = factorCollisionsIntoActions(board, ship, collision_cost)
    if vec_x != 0:
        directions[(1,0)] = vec_x * amortized_value + collision_weightings[(1,0)]
        directions[(-1,0)] = -vec_x * amortized_value + collision_weightings[(-1,0)]
    else:
        directions[(1,0)] = -amortized_value + collision_weightings[(1,0)]
        directions[(-1,0)] = -amortized_value + collision_weightings[(-1,0)]
    if vec_y != 0:
        directions[(0,1)] = vec_y * amortized_value + collision_weightings[(0,1)]
        directions[(0,-1)] = -vec_y * amortized_value + collision_weightings[(0,-1)]
    else:
        directions[(0,1)] = -amortized_value + collision_weightings[(0,1)]
        directions[(0,-1)] = -amortized_value + collision_weightings[(0,-1)]
    #staying still is divided by an additional 2 to account for the additional vec_x/vec_y weighting
    #If end.x-start.x equaled end.y-start.y then  it should be 1/2
    if not (ATTACKING_SHIPS[ship.id] and board.cells[start].halite > 0): # staying still while attacking mines halite
        directions[(0,0)] = (board.cells[start].halite/4/2 if can_mine else 0) + collision_weightings[(0,0)]

    return ( sorted(directions,key= lambda k: directions[k], reverse=True), directions, collision_weightings)
def assignProtectors(board, new_ship_avalue):
    PROTECT = defaultdict(lambda: None)
    if len(board.current_player.ships) >= 16:
        for shipyard in board.current_player.shipyards:
            protector = board.cells[shipyard.position].ship
            PROTECT[shipyard.id] = protector.id if protector else None
    protect_log = {'nsv_a': new_ship_avalue, 'Gamma': Gamma(board.step, new_ship_avalue)}
    return (PROTECT, protect_log)
def assignMovesToShips(board, order, targets, spawned_points, new_ship_avalue, PROTECT):
    global RETURNING, DANGER
    size = board.configuration.size
    def assignMove(ship_id):
        ship = board.ships[ship_id]
        target = Point(targets['attack'][ship.id]['point'][0],targets['attack'][ship.id]['point'][1]) if ATTACKING_SHIPS[ship.id] else targets['mine'][ship.id]['point']
        val = targets['attack'][ship.id]['value'] if ATTACKING_SHIPS[ship.id] else targets['mine'][ship.id]['value']

        if ship.position == target and space_taken[ship.position] != False and space_taken[ship.position] != 'spawn'\
                and ATTACKING_SHIPS[ space_taken[ship.position]['id'] ] == ATTACKING_SHIPS[ ship.id ]:
            target = space_taken[ship.position]['target']
        (actions, values, collisions) = findDesiredAction(board, ship, target, val, can_mine = not RETURNING[ship_id] and not ATTACKING_SHIPS[ship_id])
        action_copy = actions.copy()
        ship_point = ship.position
        action_dict['directions'][ship_id] = remap_keys(values)
        action_dict['collisions'][ship_id] = remap_keys(collisions)
        while True:
            top_direction = actions.pop(0)
            new_point = Point( (ship_point.x + top_direction[0]) % size, (ship_point.y + top_direction[1])%size )
            if not space_taken[new_point] or board.step > 380:
                break
            if space_taken[new_point] == 'spawn': #Undo the spawning of the ship
                board.cells[new_point].shipyard.next_action = None
                break
            if len(actions) == 0: #If this happens we are fucked, two friendly ships are colliding
                Fixed = False
                for force_dir in action_copy:
                    force_point = Point( (ship_point.x + force_dir[0]) % size, (ship_point.y + force_dir[1])%size )
                    alternative = space_taken[force_point]['alternative']
                    #If a blocking ship can move somewhere else
                    if alternative['point'] and not space_taken[ alternative['point'] ]:
                        Fixed = True
                        #Move the blocking ship to the alternative spot
                        blocking_ship = board.ships[ space_taken[force_point]['id'] ]
                        blocking_ship.next_action = DIR_TO_ACTION[ alternative['dir'] ]
                        alt_target = targets['attack'][blocking_ship.id]['point'] if ATTACKING_SHIPS[blocking_ship.id] else targets['mine'][blocking_ship.id]['point']
                        space_taken[ alternative['point'] ] = {'target':alt_target, 'id':blocking_ship.id, 'alternative': {'point': None, 'dir': None}}
                        #Set up the current ship to move into the blocking ship position
                        new_point = force_point
                        top_direction = force_dir
                        break
                break
        #If there are more options than the chosen one left
        if len(actions) > 0:
            alt_dir = actions.pop(0)
            alt_point = Point( (ship_point.x + alt_dir[0]) % size, (ship_point.y + alt_dir[1])%size )
        else:
            alt_dir, alt_point = None, None
        space_taken[new_point] = {'target':target, 'id':ship.id, 'alternative': {'point': alt_point, 'dir': alt_dir}}
        ship.next_action = DIR_TO_ACTION[top_direction]
        action_dict['actions'][ship_id] = top_direction
        global FUTURE_DROPOFF
        h = board.current_player.halite + ship.halite
        if (FUTURE_DROPOFF and ship.position == FUTURE_DROPOFF) or (BEST_NEW_DROPOFF and ship.position == BEST_NEW_DROPOFF) and h >= 500:
            ship.next_action = ShipAction.CONVERT
            FUTURE_DROPOFF = None
            space_taken[new_point] = False
            targets['mine'][ship_id] = {'point':ship.position, 'value':1, 'halite':0, 'mining_time':0, 'mined': 0, 'next_val':0}
            space_taken[ship_point] = {'target': targets['mine'][ship_id], 'id': ship.id, 'alternative': {'point':None, 'dir': None}}
            action_dict['actions'][ship_id] = 'convert'

        elif RETURNING[ship_id]: #reset returning for the ship if false
            for shipyard in board.current_player.shipyards:
                if new_point == shipyard.position:
                    RETURNING[ship_id] = False
    space_taken = spawned_points
    action_dict = {'actions':{}, 'directions':{}, 'collisions': {}}
    for ship_id in order:
        protector = False
        for shipyard in board.current_player.shipyards:
            if PROTECT[shipyard.id] == ship_id and PROTECT[shipyard.id]:
                protector = True
        if not protector:
            assignMove(ship_id)
    for shipyard in board.current_player.shipyards:
        if space_taken[shipyard.position] and PROTECT[shipyard.id]:
            assignMove(PROTECT[shipyard.id])
    return action_dict

def findAllNearestDists(board):
    global NEAREST_DROPOFF, CENTER, CENTER_VAL
    NEAREST_DROPOFF = {}
    CENTER, CENTER_VAL = None, float('inf')
    size = board.configuration.size
    for x in range(size):
        for y in range(size):
            square = Point(x,y)
            nearestDropoff(board, square)

def returnMiningShips(board, targets, nsv, dominance_map, targeted):
    global RETURNING, shipyard_occupied, STORED_HALITE_VALUE, STORED_HALITE_INCR, FUTURE_DROPOFF
    def savedTurnValue(requirement, ship, carried_halite, new_ship_avalue):
        def turns_needed(needed_halite):
            if needed_halite == 0:
                return 0
            min_turns_to_fund_ship = float('inf')
            for o_ship in board.current_player.ships:
                if o_ship.id != ship.id and not RETURNING[o_ship.id] and not ATTACKING_SHIPS[o_ship.id] and targets[o_ship.id]['value'] > 0:
                    turns_mining = np.ceil( max(needed_halite - o_ship.halite, 0) / targets[o_ship.id]['value'] )
                    return_from_spot = targets[o_ship.id]['point'] if turns_mining > 0 else o_ship.position
                    turns_to_fund_ship =  turns_mining + nearestDropoff(board, return_from_spot)['dist']
                    min_turns_to_fund_ship = min(min_turns_to_fund_ship, turns_to_fund_ship)
            return min_turns_to_fund_ship
        #First finding expected turn of the ship returnining
        needed_halite_w_return = max(requirement - carried_halite, 0)
        returning_turns = turns_needed(needed_halite_w_return)
        staying_turns = turns_needed(requirement)
        needed_halite_w_mine = max(requirement - targets[ship.id]['mined'], 0)
        mining_turns = max( turns_needed(needed_halite_w_mine), 0, targets[ship.id]['mining_time'] + nearestDropoff(board, targets[ship.id]['point'])['dist'])
        log = {'return':returning_turns, 'stay':staying_turns, 'mine':mining_turns}

        turn_value= ( Gamma(board.step + returning_turns, new_ship_avalue) - Gamma(board.step + min(staying_turns, mining_turns), new_ship_avalue) )
        return (turn_value/2, log)
    sorted_halite_order = sorted(board.current_player.ships, key= lambda k: k.halite, reverse=True)
    curr_halite = board.current_player.halite
    for ship in board.current_player.ships:
        if RETURNING[ship.id]:
            curr_halite += ship.halite
    remainder = curr_halite % 500
    i = curr_halite // 500
    dropoff_targets, dropoff_target_list = {}, {}
    ships_still_mining = []
    turn_log = {}
    dropoff_log = {}
    for ship in sorted_halite_order:
        if ATTACKING_SHIPS[ship.id]:
            continue
        #cost of returning to dropoff is the amortized value * number of turns spent returning to base
        nearest_dropoff = nearestDropoff(board, ship.position, h=ship.halite)
        nearest_dropoff_dist = nearest_dropoff['dist']
        if nearest_dropoff['point'] and board.cells[nearest_dropoff['point']].shipyard != None:
            while shipyard_occupied[ board.cells[ nearest_dropoff['point'] ].shipyard.id ][ board.step + nearest_dropoff_dist ]:
                nearest_dropoff_dist += 1

        return_value = 0
        halite = ship.halite
        while halite > 0 and i < len(nsv):
            if halite >= 500-remainder:
                halite -= 500-remainder
                #there is enough halite to fully pay for it
                return_value += Gamma(board.step + nearest_dropoff_dist, nsv[i]) * (500-remainder)/500
                remainder = 0
                i += 1
            else:
                (saved_value, turn_log) = savedTurnValue(500-remainder, ship, halite, nsv[i])
                return_value += saved_value #nsv[i] * halite / 500
                remainder += halite
                halite = 0
        return_value = return_value / nearest_dropoff_dist / 2 if nearest_dropoff_dist > 0 else 0
        mine_return_cost = targets[ship.id]['mined']/500 * Gamma(board.step + nearest_dropoff_dist + targets[ship.id]['mining_time'], nsv[i+1]) if len(nsv) > (i+1) else 0
        return_cost = max(targets[ship.id]['value'], mine_return_cost)
        nsv_i = nsv[i] if i < len(nsv) else 0
        dropoff_log[ship.id] = {'value': return_value, 'cost': return_cost, 'a_val':targets[ship.id]['value'] ,'curr_H':curr_halite, 'H':ship.halite, 'turns_saved':turn_log, 'create_first':createFirstDropoff(board, targets) }

        heavy_return_value = STORED_HALITE_VALUE * ship.halite / nearest_dropoff_dist if nearest_dropoff_dist > 0 else 0
        end_game_return = board.step > 380
        danger_return = True if sum(DANGER[ship.id]) >= 6 and ship.halite > 0 else False
        if danger_return:
            STORED_HALITE_VALUE = min(MAX_STORED_HALITE_VALUE, STORED_HALITE_VALUE + STORED_HALITE_INCR)
        n_yards = len(board.current_player.shipyards)
        if ( (n_yards > 0 and (return_value > return_cost or heavy_return_value > return_cost or RETURNING[ship.id])) or ( n_yards == 0 and createFirstDropoff(board, targets) )\
                or end_game_return or danger_return) and nearest_dropoff['point'] != None:
            #reassign target of the ship
            dropoff_targets[ship.id] = {'point':nearest_dropoff['point'], 'value': 1, 'halite': 0, 'mining_time': 0, 'mined':0, 'next_val': targets[ship.id]['next_val']}
            dropoff_target_list[ship.id] = 'returning to dropoff'
            RETURNING[ship.id] = True
            nearest_shipyard = board.cells[nearest_dropoff['point']].shipyard
            if nearest_shipyard:
                shipyard_occupied[ nearest_shipyard.id  ][board.step + nearest_dropoff_dist] = ship.id
            #if the dropoff we are heading to is the temporary one turn it into a permanent one
            #otherwise, it will be overwritten next turn
            if BEST_NEW_DROPOFF and nearest_dropoff['point'] == BEST_NEW_DROPOFF:
                FUTURE_DROPOFF = BEST_NEW_DROPOFF
        else: #ship isn't returning
            ships_still_mining.append(ship)
            RETURNING[ship.id] = False
    (targets, target_list, _, _, _) = miningLogic(board, ships_still_mining, dominance_map, targeted)
    targets.update(dropoff_targets)
    target_list.update(dropoff_target_list)
    return (targets, target_list, dropoff_log)


def nearestDropoff(board, ship_point, h=0):
    global NEAREST_DROPOFF, BEST_NEW_DROPOFF, FUTURE_DROPOFF, CENTER, CENTER_VAL
    if (ship_point, h, BEST_NEW_DROPOFF, FUTURE_DROPOFF) in NEAREST_DROPOFF:
        return NEAREST_DROPOFF[ (ship_point, h, BEST_NEW_DROPOFF, FUTURE_DROPOFF) ]
    size = board.configuration.size
    min_distance, min_pos, orig_dist = float('inf'), None, 0
    #only look at future dropoffs if we have enough money to fund them
    if FUTURE_DROPOFF and board.current_player.halite + h >= 500:
        future_dropoff_list = [FUTURE_DROPOFF]
    elif BEST_NEW_DROPOFF and board.current_player.halite + h >= 500:
        future_dropoff_list = [BEST_NEW_DROPOFF]
    else:
        future_dropoff_list = []
    shipyard_positions = [shipyard.position for shipyard in board.current_player.shipyards]
    dist_tot = 0
    for yard in shipyard_positions + future_dropoff_list:
        x_dist = abs(ship_point.x - yard.x)
        y_dist = abs(ship_point.y - yard.y)
        #length of traveling wrapping around the sides is the loop size - the forward facing path
        distance = min(x_dist, size-x_dist) + min(y_dist, size-y_dist)
        dist_tot += distance
        orig_dist = distance
        shipyard = board.cells[ yard ].shipyard
        if shipyard:
            while shipyard_occupied[ shipyard ][board.step + distance]:
                distance += 1
        if distance < min_distance:
            min_distance = distance
            min_pos = yard
    NEAREST_DROPOFF[ (ship_point, h, BEST_NEW_DROPOFF, FUTURE_DROPOFF) ] = {'dist':min_distance, 'point': min_pos, 'orig_dist': orig_dist}

    if CENTER_VAL > dist_tot: #find the center point by the summed lowest distance to all dropoffs
        CENTER_VAL = dist_tot
        CENTER = ship_point
    return {'dist':min_distance, 'point': min_pos, 'orig_dist': orig_dist}
NEAREST_EDROPOFF = {}
def nearestEDropoff(board, square):
    global NEAREST_EDROPOFF
    if square in NEAREST_EDROPOFF:
        return NEAREST_EDROPOFF[ square ]

    size = board.configuration.size
    min_distance = float('inf')
    for opponent in board.opponents:
        for e_yard in opponent.shipyards:
            distance = manhattan_distance(square, e_yard.position)
            min_distance = min(distance, min_distance)

    NEAREST_EDROPOFF[ square ] = min_distance
    return min_distance
def newMiningShipValue(board, init, general_dominance_map, augmented, assigned):
    a_list = findAmortizedValueList(board, init, dominance=general_dominance_map, discount_distance=False)
    tot_aval = sum([v['value'] for v in a_list[:5]])
    idx = 0
    while True:
        top = a_list[idx]
        _next = a_list[idx+1]
        top_val = augmented[top['point']] if augmented[top['point']] else top['value']
        next_val = augmented[_next['point']] if augmented[_next['point']] else _next['value']
        idx += 1
        if (not assigned[top['point']] and next_val <= top_val) or idx >= len(a_list) - 1:
            break
    new_mining_val = (top_val + next_val)/2
    return (top['point'], new_mining_val)
def SpawnShips2(board, augmented, assigned, general_dominance_map, attack_point_vals):
    global shipyard_occupied
    log = {'spawns':[], 'nsv':[]}
    nsv = [ gamma_pdf(board.step)]
    spawned_points = defaultdict(lambda: False)
    SHIP_MIN_VAL = 375 #600 instead of 500 assuming some of the 500 comes from friendly ships
    tot_aval = 0
    curr_halite = board.current_player.halite
    for ship_ind in range(len(board.current_player.shipyards)-1,-1,-1):
        shipyard = board.current_player.shipyards[ship_ind]
        next_ship_value = SHIP_MIN_VAL
        count = 0
        a_list = findAmortizedValueList(board, shipyard.position, dominance=general_dominance_map, discount_distance=False)
        tot_aval = sum([v['value'] for v in a_list[:20]])
        new_attack_vals = shipAttackValue(board, shipyard.position, attack_point_vals)
        (attack_target, attack_val) = bestAttackTarget(new_attack_vals, assigned)
        while next_ship_value >= SHIP_MIN_VAL and len(nsv)< 10:
            idx = 0
            while True:
                top = a_list[idx]
                _next = a_list[idx+1]
                top_val = augmented[top['point']] if augmented[top['point']] else top['value']
                next_val = augmented[_next['point']] if augmented[_next['point']] else _next['value']
                idx += 1
                if (not assigned[top['point']] and next_val <= top_val) or idx >= len(a_list) - 1:
                    break
            top_val = max(top_val, attack_val)
            GAMMA_SHIPS = min (1, 13 / len(board.current_player.ships) if len(board.current_player.ships) > 0 else float('inf'))
            #if 0 ships, we need to make 1
            next_ship_value = Gamma(board.step, top_val) * GAMMA_SHIPS
            if next_ship_value >= SHIP_MIN_VAL and board.step < 280:
                shipyard_occupied[shipyard.id][board.step + count] = 'new_ship'
            assigned[top['point']] = True
            nsv.append(top_val)
            if (next_ship_value > SHIP_MIN_VAL or board.step < 280) and not assigned[shipyard.position] and curr_halite >= 500\
                        and len(board.current_player.ships) < MAX_SHIPS and board.step < 280:
                shipyard.next_action = ShipyardAction.SPAWN
                spawned_points[shipyard.position] = 'spawn'
                curr_halite -= 500
            count += 1
        log['spawns'].append({'shipyard':shipyard.id ,'ship_val': next_ship_value, 'new_attack_val':attack_val ,'a_val': top_val})
    nsv = sorted(nsv, reverse=True)
    if len(nsv) > 0:
        log['nsv'] = {'local':nsv, 'tot': tot_aval}

    return (nsv, spawned_points,log)
GAMMA_VALS = [0]*401
def Gamma(turn, actual):
    #calculates how much better we are doing than anticipated
    difference = max(0, actual / GOLDEN_RATIO - gamma_pdf(turn) )
    #add the difference to every further step
    return OGamma(turn) + difference * (MAX_STEPS - turn)
def OGamma(turn):
    if turn > SHIP_SPAWN_LIMIT:
        return 0
    global GAMMA_VALS
    if GAMMA_VALS[0] == 0:
        tot = 0
        for t in range(SHIP_SPAWN_LIMIT,0,-1):
            v_t = gamma_pdf(t)
            tot += v_t
            GAMMA_VALS[t] = tot
        GAMMA_VALS[0] = GAMMA_VALS[1]
    return GAMMA_VALS[int(turn)]
def gamma_pdf(t):
    t = 1 if t == 0 else t
    if t > 120:
        v_t = GAMMA2_TUNE * np.log(t) + GAMMA2_CONST
    else:
        v_t = GAMMA1_TUNE * np.log(t) + GAMMA1_CONST
    return v_t / GOLDEN_RATIO
def createFirstDropoff(board, targets):
    total_amortized_value = 0
    n = max( len(board.current_player.ships), 1)
    for ship in board.current_player.ships:
        if not ATTACKING_SHIPS[ship.id]:
            total_amortized_value += targets[ship.id]['value']
    if board.step < 300:
        return True
    #always returns True for now
    return True if total_amortized_value + n * OGamma(board.step) > 1000 else True
def manhattan_distance(start, end):
    size = 21
    x_dist = abs(end.x - start.x)
    y_dist = abs(end.y - start.y)
    return min(x_dist, size-x_dist) + min(y_dist, size-y_dist)
def squareDistance(start, end):
    SQUARE_MAX = 5
    size = 21
    x_dist = abs(end.x - start.x)
    y_dist = abs(end.y - start.y)
    return min(x_dist, size-x_dist)  <= SQUARE_MAX and min(y_dist, size-y_dist) <= SQUARE_MAX
def decideIfCreateDropoff(board, ships, targets, attacking_ships, assigned, general_dominance_map):
    global BEST_NEW_DROPOFF
    n_yards = len(board.current_player.shipyards)
    n_ships = len(board.current_player.ships)
    if n_yards > n_ships / MAX_SHIP_TO_YARD_RATIO:
        BEST_NEW_DROPOFF = None
        return (None, {'savings':0}, None, {'harbor': None, 'mhv': 0})
    #if we run the function to find new dropoffs, reset center
    size = board.configuration.size
    def dropoffSavings(square, ships, harbor_value, harbor_dist):
        total_saved = 0
        print_total_saved = ""
        for ship in ships:
            ship_point = ship.position
            square_distance = manhattan_distance(ship_point, square)
            nearest_dropoff = nearestDropoff(board, ship_point, h=ship.halite)
            nearest_dropoff_dist =nearest_dropoff['dist']
            if nearest_dropoff['point'] and board.cells[nearest_dropoff['point']].shipyard != None:
                while shipyard_occupied[ board.cells[ nearest_dropoff['point'] ].shipyard.id ][ board.step + nearest_dropoff_dist ]:
                    nearest_dropoff_dist += 1
            #was next_val
            saved_amount = targets[ship.id]['value'] * max(nearest_dropoff_dist - square_distance,0)
            #we can't save more than our amortized value * turns left in game
            saved_amount = max(saved_amount, targets[ship.id]['value'] * (MAX_STEPS - board.step ) )
            total_saved += saved_amount

        #We save the total amortized value of the harbor * 1 + STORED HALITE VALUE
        #This is saying we would have to return once and again with the increasing return rate
        total_saved += harbor_dist * harbor_value/DROPOFF_LIMIT * (1+ STORED_HALITE_VALUE)

        return (total_saved, print_total_saved)
    def harborSavings(square):
        amortized_list = findAmortizedValueList(board, square, max_dist=DROPOFF_VISION, store=False, discount_distance=False)[:DROPOFF_LIMIT]
        harbor_sum = sum( [a_val['value'] if a_val['point']!=square else 0 for a_val in amortized_list] )
        return harbor_sum
    def attackerSavings(square):
        total_dist, n = 0,0
        for ship in attacking_ships:
            if RETURNING[ship.id]:
                dist = manhattan_distance(ship.position, square)
                prev_dist = nearestDropoff(board, ship.position, ship.halite)['dist']
                total_dist += max(0, dist-prev_dist)
                n += 1
        return total_dist / n if n > 0 else 0
    def dangerous(square):
        for (x_move, y_move) in MOVES:
            adj_square = Point( (square.x + x_move)%size, (square.y + y_move)%size )
            adj_ship = board.cells[adj_square].ship
            if adj_ship and adj_ship.player_id != board.current_player:
                return True
        return False
    max_savings, best_dropoff = 0, None
    convert_savings, best_convert = 0, None
    max_print = ""
    mhv, best_harbor, best_harbor_dist, ohv = float('-inf'), None, 0, 0
    tested = defaultdict(lambda: {'v':0, 'n':0})
    all_points = {}
    for ship in ships:
        for point_x in range(board.configuration.size):
            for point_y in range(board.configuration.size):
                square = Point(point_x, point_y)
                if assigned[square] or dangerous(square):
                    continue
                dist = manhattan_distance(ship.position, square)
                if dist > DISTANCE_THRESHOLD:
                    continue
                nearest_dropoff_dist = nearestDropoff(board, square, ship.halite)['dist']
                #don't assign squares that are targets as dropoffs
                if nearest_dropoff_dist < DISTANCE_BETWEEN_DROPOFFS or nearestEDropoff(board,square) < DISTANCE_BETWEEN_DROPOFFS-2:
                    continue
                #finds best point
                tested[square]['n'] += 1
                if tested[square]['n'] == SHIPS_REQUIRED_IN_DROPOFF_DISTANCE or (n_yards == 0 and tested[square]['n'] == 1):
                    if len(attacking_ships) < len(targets) or True:
                        tested[square]['v'] = harborSavings(square)
                    else:
                        tested[square]['v'] = attackerSavings(square)
                if tested[square]['n'] >= SHIPS_REQUIRED_IN_DROPOFF_DISTANCE or n_yards == 0:
                    harbor_value  = tested[square]['v']
                    #harbor value is the sum of the amortized values of things from that point - steps*a_val of the ship that needs to create the dropoff
                    if harbor_value - dist * targets[ship.id]['value'] > mhv:
                        best_harbor = square
                        mhv = harbor_value - dist * targets[ship.id]['value']
                        ohv = harbor_value
                        best_harbor_dist = nearest_dropoff_dist
    if best_harbor:
        (savings, print_total_saved) = dropoffSavings(best_harbor, ships, ohv, best_harbor_dist)
    else:
        savings, print_total_saved = 0, ''
    if (savings > 500 and n_yards <= n_ships/MAX_SHIP_TO_YARD_RATIO) or ( n_yards== 0 and not FUTURE_DROPOFF):
        BEST_NEW_DROPOFF = best_harbor
    else:
        BEST_NEW_DROPOFF = None
    return (best_harbor, {'savings':savings}, max_print, {'harbor': best_harbor, 'mhv': mhv})
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
                    move_x, move_y = move.x, move.y
                    if abs(move_x) > 1:
                        move_x = int( np.sign(move_x) * -1)
                    if abs(move_y) > 1:
                        move_y = int( np.sign(move_y) * -1)
                    vector += Point(move_x, move_y)
                elif i == STORED_MOVES - 1 and s: #the last move is from the last recorded position to the current one
                    move = ship.position - s
                    move_x, move_y = move.x, move.y
                    if abs(move_x) > 1:
                        move_x = int( np.sign(move_x) * -1)
                    if abs(move_y) > 1:
                        move_y = int( np.sign(move_y) * -1)
                    vector += Point(move_x, move_y)
                    #vector += move #add all incremental moves to create vector
            enemy_squares[ship.id][STORED_ITER] = ship.position
            enemy_vectors[ship.id] = vector
def remap_keys(mapping):
        return [{'point':k, 'value': v} for k, v in dict(mapping).items()]
PREV_SHIPYARDS = 0
def agent(obs, config):
    global logging_mode, print_log
    if logging_mode:
        start = time.time()
    size = config.size
    board = Board(obs, config)
    my = board.current_player
    ships = my.ships
    global PREV_SHIPYARDS
    if len(my.shipyards) != PREV_SHIPYARDS: #not perfect but okay (hopefully)
        findAllNearestDists(board)

    if logging_mode:
        step_log = {}
        step_log['ship_positions'] = {}
        for ship in board.current_player.ships:
            step_log['ship_positions'][ship.id] = (ship.position, ship.halite)
        step_log['enemy_ship_positions'] = {}
        for ship_id in board.ships:
            ship = board.ships[ship_id]
            if ship.player_id != board.current_player_id:
                step_log['enemy_ship_positions'][ship.id] = (ship.position, ship.halite)

    global SQUARE_AVALS, BORDERS
    SQUARE_AVALS = defaultdict(lambda: 0)
    BORDERS = {}
    updateEnemyVectors(board)

    if logging_mode:
        step_log['enemy_vectors'] = enemy_vectors.copy()
        end_setup = time.time()

    # create the dominance map
    (general_dominance_map, dominance_map) = createDominanceMap(board, ships)
    if logging_mode:
        end_dom = time.time()
    
    # setup for attacking/mining logic
    attack_ships = []
    mining_ships = []
    for ship in ships:
        if ATTACKING_SHIPS[ship.id]:
            attack_ships.append(ship)
        else:
            mining_ships.append(ship)
    
    # run attack logic
    (attack_targets, targeted, attack_point_vals, attack_order, no_targets_assigned, attack_log) = attackLogic(board, attack_ships)
    if logging_mode:
        step_log['attack_logic'] = attack_log
        end_attack = time.time()
    
    # run mining logic
    mining_ships = mining_ships + no_targets_assigned
    (mining_targets, target_list, assignment_order, augmented, assigned) = miningLogic(board, mining_ships, dominance_map, targeted)
    if logging_mode:
        end_mining = time.time()
    
    # decide whether to spawn ships
    targets = {'mine': mining_targets, 'attack': attack_targets}
    (nsv, spawned_points, spawn_log) = SpawnShips2(board, augmented, assigned, general_dominance_map, attack_point_vals)
    if logging_mode:
        step_log['spawn'] = spawn_log
        end_spawn = time.time()
    
    # decide whether to create dropoffs. Clear center of our map for next turn
    (best_dropoff, max_savings, mp, harbor) = decideIfCreateDropoff(board, mining_ships, targets['mine'], attack_ships, assigned, general_dominance_map)
    if logging_mode:
        step_log['dropoff'] = {'best_point': best_dropoff, 'value': max_savings, 'math': mp, 'harbor': harbor}
        step_log['dropoff']['future'] = FUTURE_DROPOFF
        end_dropoff = time.time()
    
    # decide whether mining ships should return
    (mine_targets, target_list, log_dropoffs) = returnMiningShips(board, targets['mine'], nsv, dominance_map, targeted)
    if logging_mode:
        targets['mine'] = mine_targets
        step_log['returns'] = log_dropoffs
        end_return = time.time()

    # assign protectors to shipyards
    new_ship_avalue = nsv[1] if len(nsv) > 1 else 0
    (protect, protect_log) = assignProtectors(board, new_ship_avalue)
    if logging_mode:
        end_protect = time.time()

    # assign moves to ships
    actions = assignMovesToShips(board, assignment_order + attack_order, targets, spawned_points, new_ship_avalue, protect)
    if logging_mode:
        step_log['ship_actions'] = actions
        step_log['protect'] = protect_log
        step_log['danger'] = DANGER.copy()
        end_assign_moves = time.time()

    # assign tasks to ships
    assign_log = assignTaskToShips(board, targets, attack_point_vals, general_dominance_map, augmented, assigned)
    if logging_mode:
        step_log['assign'] = assign_log
        end_assign_tasks = time.time()


    PREV_SHIPYARDS = len(my.shipyards)

    if logging_mode:
        global log
        for ship in dominance_map:
            dominance_map[ship] = remap_keys(dominance_map[ship])
        step_log['dominance'] = dominance_map
        step_log['mining'] = {'order': assignment_order, 'augmented': remap_keys(augmented), 'assigned': remap_keys(assigned),'targets':{}}
        for ship_id in targets['mine']:
            step_log['mining']['targets'][ship_id] = targets['mine'][ship_id]

        for ship_id in log_dropoffs:
            try:
                prev = log[-1]['returns'][ship_id]['H']
            except:
                prev = 0
            mined_t = max(log_dropoffs[ship_id]['H'] - prev, 0)
            total_mined[ship_id] = (total_mined[ship_id][0] + mined_t, total_mined[ship_id][1]+1)
        step_log['total_mined'] = dict(total_mined).copy()
        step_log['returning_list'] = RETURNING.copy()
        step_log['center'] = getBorders(board)

        end = time.time()
        step_log['time'] = end - start
        log.append(step_log)

    if logging_mode and print_log:
        log_str = "turn: {}\nattacking ships: {}\nmining ships: {}\nattack_target: {}\nhas_decimated: {}\n" \
                  "best new dropoff: {}\nfuture dropoff: {}\n" \
                  "setup time: {}\ndominance_map time: {}\nattack_logic time: {}\nmining_logic time: {}\n" \
                  "spawning_logic time: {}\ndecide_create_dropoff time: {}\ndecide_return time: {}\n" \
                  "protect time: {}\nassign_moves time: {}\nassign_tasks time: {}\nextra_logging time: {}\n"
        print(log_str.format(board.step,
                             len(attack_ships),
                             len(mining_ships),
                             OPPONENT_TO_TARGET if OPPONENT_TO_TARGET != None else 'None',
                             HAS_DECIMATED if HAS_DECIMATED >= 0 else 'None',
                             BEST_NEW_DROPOFF if BEST_NEW_DROPOFF != None else 'None',
                             FUTURE_DROPOFF if FUTURE_DROPOFF != None else 'None',
                             end_setup - start,
                             end_dom - end_setup,
                             end_attack - end_dom,
                             end_mining - end_attack,
                             end_spawn - end_mining,
                             end_dropoff - end_spawn,
                             end_return - end_dropoff,
                             end_protect - end_return,
                             end_assign_moves - end_protect,
                             end_assign_tasks - end_assign_moves,
                             end - end_assign_tasks))

        if board.step == 100:
            with open('log.txt', 'w') as log_file:
                json.dump(log, log_file)

    return my.next_actions