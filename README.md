**Pacman Search Algorithms**
Explore AI-based Search Algorithms in Pacman’s World!
This project implements A*, A* LifeLong, and D* Lite search algorithms for pathfinding in a Pacman environment. It allows you to compare algorithm performance in static and dynamic environments.

**Project Overview**
This repository is based on the research paper "D* Lite" by Sven Koenig & Maxim Likhachev.
It explores how different search algorithms behave in constrained, evolving grid environments.

**Project Structure:**

1) pacman.py → Main game logic
2) search.py → Implementation of search algorithms
3) searchAgents.py → Custom agents using A*, A* LifeLong, and D* Lite

**How to Run**
To see Pacman in action, use the following commands:

**For A* Algorithm:**

python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l contoursMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic


**For A* LifeLong Algorithm:**

python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l contoursMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic

**For D* Lite Algorithm:**

python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l contoursMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic

**Speed Up Pacman**
You can speed up Pacman by adding --frameTime 0 to the command.
To change the search strategy, modify fn to one of the following:

astar → A* Search
astarll → A* LifeLong
dstar → D* Lite

**Environment & Dependencies**
Python 3.9+

Clone the repository and install dependencies (if needed).

**References**
Research Paper: D* Lite
Pacman AI Projects: UC Berkeley
