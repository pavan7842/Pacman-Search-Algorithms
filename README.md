## LPA* and D* Lite Implementation

This repository implements the research paper **"D* Lite"** by Sven Koenig, Maxim Likhachev

In order to see pacman in action, the following are commands for six different mazes:
```
python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l contoursMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic
python pacman.py -l contoursMaze -z .5 -p SearchAgent -a fn=astarll,heuristic=manhattanHeuristic

python pacman.py -l smallMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic
python pacman.py -l contoursMaze -z .5 -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic

```
You can speed up Pacman by adding ```--frameTime 0``` and to change search strategy change ```fn``` to one of:
```
astar ===> aStarSearch
astarll ===> aStartSearchLifeLong
dstar ===> dStarSearch


Implemented using Python 3.9.7
