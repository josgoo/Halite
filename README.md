# The Bot

## (Target - Moving) Amortized Value Logic Pipeline
Similar to other teams, at a high level our logic pipeline had every ship choose a target square and then assigned each ship an optimal action to take to reach that target. To choose each target and action, our architecture computed an amortized value for every target which generally took the form

<div align="center"> $\frac{Total \ Value \ of \ Target}{Total \ Turns \ Needed} * (Probability\ of \ Success)$ </div>

## Dominance Map
In order to determine the probability of successfully reaching a given square, we needed to have some way of representing which players dominated which parts of the board. Our final dominance map consisted two components: a *global* dominance map and a *ship-specific* dominance map. For the global dominance map, we created a sphere of influence for every enemy ship. The level of influence a ship exerts on a square is proportional to the probability that the ship will enter that square at some point over the next 2 turns, given that the ship is moving completely randomly. These probabilities effectively form a plus-shaped Gaussian blur around each enemy ship, as can be seen from the diagram below. `TODO: add diagram?`. Once we had a sphere of influence for each enemy ship, we used them to calculate for each square the probability that no enemy ship would enter that square for the next 2 turns. These probabilities formed the base of our global dominance map, with higher values indicating a smaller likelihood of any enemy ship crossing a square and therefore meaning that square is safer.

For the ship-specific dominance map, we used the process as above but only factored in enemy ships lighter than the current ship (rather than all enemy ships). The final dominance map for each ship was a linear combination of the global dominance map and that ship's ship-specific dominance map. The ship-specific dominance map was weighted to be about 3 times as important as the global dominance map, with the idea that the global map provides a long-term deterrent warning friendly ships not to go to enemy dominated territory while the ship-specific map helps decide short-term collision avoidance.

## Assigning Ship Mode
The final bot allowed ships to switch between two overarching "modes": mining and attacking. All ships began as attacking ships on spawn, and at the end of every turn each ship re-evaluated whether it could attain a higher value target as a mining ship or an attacking ship. The ship would then switch to the mode promising the higher value on the next turn. Because the exact mining or attacking value of a ship depended on the actions of other friendly ships, when re-evaluating the modes each ship assumed it would be the last to act and therefore the last to choose its target on the next turn. This method effectively led to a conservative lower-bound estimation of the value of each mode.

However, we often found that the mining and attacking values were relatively close, leading to ships that would often flip flop between modes on consecutive turns. This constant switching caused ships to continually change targets, thereby making them less effective. To combat this issue, we made the modes "sticky"&mdash;for a ship currently in mining mode to switch to attacking mode, the attacking value would have to be significantly higher than the mining value (and vice versa). This stickiness allowed for more consistent long-term behavior over many turns.

## Mining
For each mining ship, we compiled a list of all squares which contained halite and computed an amortized value for each of those squares. Specifically, the amortized value of a square was the solution to the maximization problem

<div align="center"> $\max_{t} \ $<font size="+2">$ \frac{(1 \ - \ 0.75^{t}) \ * \ H}{dist \ + \ t}$</font>$*\ D$ </div>

where *H* is the amount of halite in the square, *t* is the amount of time to spend mining on the square, *dist* is the ship's Manhattan distance to the mining location, and *D* is the downweighting factor for the square given by the dominance map. Note that we did not factor in returning to base when computing these mining values at all. This choice was an active one, as we believed that having a separate model for deciding when a ship should return would serve us better.

As an interesting side note, our initial attempts at solving this maximization problem were far too slow. Even with incredible numerical libraries like `numpy` and `scikit`, maximizing a function in Python is time consuming. Our first optimization was to simply calculate the value of the expression for all integers in the range 1 to 20 and to take the integer that produced the highest value. Here, we both knew that ships could not mine for a fractional number of turns and assumed that no ship could ever feasibly mine a single spot for longer than 20 turns. However, this piece was still the bottleneck of our entire bot. Finally, after toying with the expression for a while, we realized that the *H* term disappears in the derivative of this function. This fact implied that the maximum was not based on *H*! Moreover, since *dist* is fixed for a given (ship, square) pair, the maximum value of the expression only depends on *t*. Therefore, we were able to pre-compute all values of the expression for *t*=[1, 20], and do a constant-time lookup at runtime.

Once every ships had its list of mining targets and associated amortized values, we used "loss" to assign a specific target to each ship. This term means that we determined the mining ship assignment order by how much lower a ship's amortized value would be if it wasn't assigned to its best mining target. In other words, we wanted to minimize the lost opportunity cost each ship would experience if it wasn't assigned its optimal mining target. This idea translates mathematically to

<div align="center">$loss = ship\_amortized\_value\_list[0] - ship\_amortized\_value\_list[1]$. </div>

As targets were assigned to mining ships, we removed the assigned targets from consideration and updated the amortized value lists and thereby the loss of each ship. This idea attempted to maximize the total mining value over all ships while still using a greedy algorithm.

However, it is clear that a greedy algorithm is suboptimal here. To get closer to the best possible solution, we allowed ships to look one step beyond their current best mining target. Once a ship was assigned a target $t_1$, we recomputed the mining target amortized value list for that ship, this time assuming that the ship was already on top of $t_1$ (and therefore excluding it from consideration). The best target in this new list, say $t_2$, was a proxy for where the ship would want to go after reaching $t_1$. We then discounted the value of $t_2$ in the amortized value lists of all other ships by a constant $\beta$ factor to discourage other ships to travel to $t_2$ immediately. This one-step-ahead discounting of future mining spots was an attempt to better coordinate the swarm of mining ships, spreading them out and preventing all our ships from targeting a small number of very valuable areas.

## Attacking
Similar to mining ships, for each attacking ship we compiled a list of amortized values for each square that we considered an attack target. A square was a viable attacking target if it was within a Manhattan distance of 2 from any enemy ship. The amortized attack value for each square was

<div align="center"> <font size="+2"> $\sum_{e\ \in enemy\ ships} \frac{(Probabiliy\ of\ Capture)_e\ *\ (Probability\ Actualized)_e\ *\ (b \ +\ min(500,\ cargo_e)\ )}{n_e\ *\ (dist\ +\ ctime_e\ )}  $ </font> </div>

The estimated probability that a particular enemy ship *e* would move to the square on the next turn was (Probability Actualized). From that square, we again computed the probability that the ship would move from there to each of the four immediately adjacent squares. These probabilities defined (Probability of Capture) for those squares. We defined the implicit value of destroying an enemy ship with the constant $b$. *$n_e$* was the number of friendly ships already assigned to attack enemy ship *e*, *dist* was the Manhattan distance from the friendly ship to the square, and *$ctime_e$* was the number of turns we expected it to take to successfully capture enemy ship *e*.

After computing these amortized attack value lists, we again assigned an attack target to each ship in order of ship loss (in the same fashion as mining).

## Assigning Actions to Ships: TODO
To assign specific actions to each ship from its target we once again used loss to order the actions of each ship with the exception that ships returning to our base or in danger had precedent. 

To actually determine direction, we labeled each action with the positive amortized value if it moved towards its target, negative amortized value if it moved away, and ( 0 + value of mining ) for staying still. Then we added a collision weighting onto the values and chose the direction with the highest value.

We also ran into cases where our ships wanted to move through eachother or would get trapped. In these cases, if the ships were of the same type and neither were returning, we swapped the targets for each. 

## Collision Avoidance: TODO
To implement collision avoidance for our bot, we needed some way to anticipate what the opponent ships were doing. To do so, we stored the previous 3 moves each ship had made, and created a vector from it. This enemy vector gave us some idea of where the ship was planning on going. We purposely kept the enemy move prediction very general to not overfit it to different playstyles. We weren't confident we could do a more involved prediction analysis without accounting for the enemy strategy. We combined the vector with an uncertainty discount blur to give some amount of weight to every direction. 

From the enemy movement prediction, we added a collision cost to each square.

<div align="center"> $(Collision\ Coef)\ *\ (Collision\ Cost)\ *\ (Probability\ of\ Move) $ </div>
where the (Probability of Move) was the vector probability an enemy ship moved to the square in question. The cost of collision was
<div align="center"> $ (Collision\ Cost) = 500\ +\ cargo\ +\ (Amortized\ Value)\ -\ (Expected\ Amoritzed\ Value\ of\ New\ Ship)$ </div>
The (Collision Coef) was the number of enemy ships with less cargo from that player surrounding our ship. The idea behind it being players with more ships surrounding our ship were more of a threat given that they can better coordinate their ships to attack more effectively. 

## Returning To Base: TODO
Our returning to base strategy was coded as an independent thought from each ship's other options. Our attacking ships simply returned if they had any cargo. It is important to note, an attacking ship with cargo could also transition to a mining ship if it was better to mine than return. Our mining ships only returned if at least one of the following requirements were met: the ship was in danger, the ship was too heavy, the game was about to end, or the ship's cargo would speed up the number of turns it would take to create another ship.

We crudely assed danger on a per ship basis through each ship's distance to lighter enemy ships over the past few turns. Every lighter enemy ship 1 manhattan distance away added 2 "danger" and every lighter empty ship 2 manhattan distance away added 1 "danger." A ship was considered in danger if the sum of the "danger" over the past 3 turns exceeded 5.

A ship was considered heavy based on a global danger variable. Our global danger value started near zero and every time any ship was in danger, the global value increased. The more global danger there was, the more incentivized our ships would be to return. The implementaion of this consisted of giving the nearest dropoff from a ship the value of $\frac{(Global\ Danger\ Value)\ *\ Cargo}{dist}$. If this value was greater than the amortized value of mining, the ship would return.

We defined the endgame for each ship to be step $400 - dist - (buffer)$ where buffer was some additional time for ships to return to base. Any ship with cargo past their endgame step would be forced to return.

Lastly, in the beginning of the game we wanted ships to return more frequently to fund the creation of new ships. The return value for these ships was $\frac{(New\ Ship\ Value) * cargo/500}{dist}$ 
$\textbf{Left out our gamma garbage}$

## Dropoff Spawning: TODO
Our decision of where to spawn dropoffs was one of the most difficult problems we needed to solve. Unlike other sections, we weren't able to quantitatively figure out what truly made a dropoff position better than another besides its proximity to good halite sections. On a high level, a well positioned dropoff not only acted as a hub for ships to be created from and return to, but it was also a leading factor in board control. Often in our games, other team's dropoff positioning directly led to their domination of the board and prevented ours.

Our dropoff algorithm also focused on providing a hub for mining and not necessarily attacking ships. Due to this, our attacking ships would frequently kill an enemy ship and then proceed to not be able to successfully return to our dropoff before also getting picked off and destroyed.

To determine where we put our dropoffs, we looked at every square that met a list of requirements and choose the best one. A potential dropoff square needed to be within a certain distance from at least 2 of our friendly ships and 1 friendly dropoff, at least a certain distance away from any friendly/enemy dropoff, and not next to a 0 cost enemy ship. These conditions existed just so that we could reasonably protect the new shipyard if needed.

Of these potential locations, we summed up the top D amortized values of mining spots near the potential location. That is, if we created D new mining ships, what would the sum of their amortized values be right now. The square with the largest sum amortized value was the spot we chose. 

Once that square was chosen, we determined whether or not we should actually build that dropoff in the future. To do so, we just calculated every ship's current total savings from the introduction of the new shipyard. Each ship theoretically could save its amortized value * how many fewer turns it needs to take to now return.
<div align="center"> $ Savings = \sum_{s \in ships} \ (Amortized\ Value)_s\ *\ max(\ 0,\ dist_s - new\_dist_s\ )$ </div>

If the new dropoffs savings exceeded 500, we saved that location as a future dropoff. The first ship in the future to return to it would then create that specific shipyard.

 
    
## Ship Spawning: TODO
We originally created a model to determine the expected returns of a mining ship over time. That is, if we created a ship on turn $t$, how much halite would we expect it to mine and successfully return with. Unfortunately, with the advent of attacking ships and the varying farming strategies of opponents, this metric was often incorrect. 

We were unable to mathematically model the value of a ship over time because of that. Instead, we chose to always spawn ships up to a somewhat arbitrarily determined turn if we could. 
    

### Shipyard Protection: TODO
In watching our games, we found a large discrepancy in our games if an enemy destroyed our shipyards early in the game or not. Losing a shipyard was a huge blow; however, spending a ship to stay on the shipyard and continuously protect it was also a large cost. Replacing a shipyard cost 1000 halite (convert a ship + replace the ship) while protecting cost 500 per ship. If an enemy trades one of their ships for one of our protecting ships, than it still not only cost us 1000 halite to protect the shipyard (500 for the first ship + 500 for the second protecting ship), but it also cost us the oppertunity cost of having the ship go out and mine/attack!

To us, this originally meant defending a shipyard was only worth it if someone would attack us without protection, but wouldn't with protection.

In our imperical findings though, a shipyard ended up being worth more than 1000 halite. It provided an unquantified amount of board control for our ships. Because of this, we once again didn't come up with a completely sound solution. We chose to have a protector for every shipyard once we had reached a certain number of ships with the idea being that the decreasing marginal value of each ship would at some point be below that of spending a ship to defend our base. We chose that number to be 15 through trial and error.

We did implement a slightly more involved protection where we would leave a protector on a shipyard if there was an enemy ship nearby and we were within a manhattan distance of 1 from the shipyard. We didn't want to mess with out amortized value system too much.


# Approaching a Halite Problem: What We Learned

While we learned a lot from thinking through and solving the many individual smaller-scale problems that formed the basis of our Halite bot, perhaps the most important things we discovered were lessons on forming high-level problem-solving approaches for large, complex problems like Halite. Before Halite IV, no one on the team had had any experience with coding competitions, and there are a handful of key takeaways which we plan to use when tackling future computer science projects. Hopefully they will provide some guidance to other first-time coding competitors, helping them avoid the biggest issues we ran into.

## Take time to plan a modular, extensible solution framework.
When our team first read the game rules of Halite, we had a single immediate thought: "*this game is all about mining!*". One team member brought up the potential idea of attacking, but after a short discussion we wrote it off as useless. After all, attacking is risky and would therefore surely be a net negative for our bot&mdash;any opponent with a half-decent collision avoidance mechanism would be uncatchable, right? With that settled, we got to work theorizing and optimizing the perfect mining bot. For the first six weeks of the competition, not a single line of code was written. We ensured that every aspect of our design was sound, solved the edge cases that needed solving, and wrote pseudocode before even thinking of opening our IDEs to avoid the traps and flaws often created by short-sighted decision making. And yet, after all that careful planing, by the final weeks of the competition our code base was a tangled mess full of baffling logic that supposedly used to make sense, leaving us wondering where everything went wrong.

As can be seen from the above anecdote, our initial bot was designed to mine. Everything about the project architecture and code was designed to allow for a modular, extensible mining bot. However, we later realized that contrary to our initial assessment, our bot should be able to attack enemy ships. And prioritize attack targets. And defend shipyards. And help friendly ships. And chase away mining enemy ships. And camp enemy shipyards. And do a hundred other things that we never originally considered. How do you incorporate a hundred new strategies into a bot that is designed to mine?

The short answer is, you don't. Shoehorning an attack strategy into our modular, extensible mining strategy required us to throw our thoughtful design work out the window, and adding more strategies became exponentially more difficult. The final result was a series of hacky workarounds and hard-coded special casework that made it nearly impossible to trace through and understand why our bot was doing what it was doing. The lesson here is to challenge your problem assumptions from the very beginning, and to take extra time to understand the problem you are facing from various angles, beyond just the first way that seems to make sense. Moreover, trying to abstract your solution framework to allow the entire thing to be modular and extensible is incredibly useful. Had we initially done this abstraction, we would have realized that mining is just one potential "mission" that a ship can take on, and that there may be other useful missions too (team "KhaVo Dan Gilles Robga Tung" has a great description of this kind of solution framework in [their writeup](https://www.kaggle.com/c/halite/discussion/183312)). This framework would have allowed missions to be modular and extensible, making it easy to add new missions (like attacking or defending) later on.

## Let data drive decisions, and design with data collection in mind from the start.
There is a saying which goes, ["collecting data is cheap, but not having it when you need it can be expensive."](https://www.datadoghq.com/blog/monitoring-101-collecting-data/) Halite IV made the truth of this quote very clear to us. We never had a good way of collecting data about our bot's decision-making process, and therefore did not have a holistic way of analyzing games and determining what needed to be improved. This lack of a data collection and analysis pipeline meant that we had to watch, evaluate, and interpret our own game replays by hand (by this point, we expect that all the data scientists reading this writeup have fainted&mdash;apologies in advance). This manual process made it tedious and tough to spot weird inconsistencies and bugs in the first place, but even worse, it meant that we were basing our changes on evidence with a sample size of 1. Such micro-level decision making inevitably causes misinterpretations and leads to "fixing bugs" without recognizing larger overarching issues. It also made it basically impossible to accurately evaluate the effects of new features or to productively tune hyperparameters. We found that by the end of the competition, many tuned versions of almost identical bots ended up with similar elo ratings, suggesting that the time spent tuning was not useful and could have been spent improving other aspects of the bot. A proper logging and analysis system would have prevented these headaches, and we believe our inability to let data drive decisions near the finale was the barrier that kept us from achieving a higher placement.

That said, it is important to acknowledge that we actively decided not to build any data tooling up front. Given this competition was our first, we were not sure that the high fixed cost of building these tools would pay off, since we were not sure how well we would perform. Interestingly, we managed to get pretty far without any data analysis at all; we built our first few bots without looking at any leaderboard games, and they peaked around the top 100. Once our bots were competing at a higher level, however, the necessity of data tooling became clear, and we struggled and eventually failed to break the top 25.