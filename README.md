# TicTacToe AI - A Battle Between AI/ML Algorithms
All I wanted to see the battle between AI/ML algorithms in the field of TicTacToe! Primary plan is to implement TicTacToe and some AI/ML algorithms like Minimax, Logistic Regression, CNN and see who they fight with each other. Shall we begin?

Wait! Check TIcTacToe AI - A Battle Between AI Algorithms.ipynb for full implementation and output: https://github.com/hillolkallol/TicTacToe-AI-A-Battle-Between-AI-Algorithms/blob/master/TicTacToe%20AI%20-%20A%20Battle%20Between%20AI%20Algorithms.ipynb

Now, it's time to start!

## What is TicTacToe?
___________________________________________________________________________________________________________
TicTacToe is a simple board game: https://www.youtube.com/watch?v=5SdW0_wTX5c

Caution: Just for fun!! Haven't captured a few edge cases (i.e. wrong user input, index out of bound) Will be fixed in the next release! :p :p

Wiki: Tic-tac-toe (American English), noughts and crosses (Commonwealth English), or Xs and Os, is a paper-and-pencil game for two players, X and O, who take turns marking the spaces in a 3×3 grid. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner. It is a solved game with a forced draw assuming best play from both players.

Reference: Wikipedia- https://en.wikipedia.org/wiki/Tic-tac-toe
___________________________________________________________________________________________________________

## TicTacToe Game Implementation
This game has two main characteristics- Move and Winner Check. CheckWinner method will be called after each move.

Before executing move, validMove method will be called from the client side just to check if the move is valid. If the move is valid, the move method will be called. The sign will be planted in the board (based on given row and col) Then the checkWinner method will be called. Last thing we need to check is if the match is a draw!

Next thing is to check if the player is the winner after the last move. Check-

* if all the rows contain same sign OR
* if all the columns contain same sign OR
* if all diagonal or anti-diagonal boxes contain same sign

### Next Move
Before executing move, validMove method will be called from the client side just to check if the move is valid.
If the move is valid, the move method will be called.
The sign will be planted in the board (based on given row and col)
Then the checkWinner method will be called.
Last thing we need to check is if the match is a draw!

```python
    def move (self, r, c, player):
        if not self.isValidMove(r, c):
            return -1
        
        sign = self.players[player-1]
        self.board[r][c] = sign
        
        # self.generateNewRow()
        self.moves.append([r, c])
        
        # self.displayBoard()
        
        if self.checkWinner (r, c, player):
            # print("Player ", player, " wins!")
            return 1
        
        if self.checkDraw ():
            # print("Draw!")
            return 0
        
        # print("Next move please!")
        return 2
```
### Check if the Match is a Draw
Check if the game is a draw. Check if the board is full.

```python
    def checkDraw (self):
        status = len(self.moves) == self.size * self.size
        if status:
            self.setWinner(0)
            
        return status
```

### Check if there is a Winner
Check if the player is the winner after the last move.
Check-
* if all the rows contain same sign OR
* if all the columns contain same sign OR
* if all diagonal or anti-diagonal boxes contain same sign

```python
    def checkWinner (self, r, c, player):
        status = self.checkRow (r, player) or self.checkCol (c, player) or self.checkDiagonals (player)
        if status:
            self.setWinner(player)
        return status
        
    def checkRow (self, r, player):
        for i in range (self.size):
            if not self.board[r][i] == self.players[player-1]: 
                return False
        
        # print("Row true")
        return True
        
    def checkCol (self, c, player):
        for i in range (self.size):
            if not self.board[i][c] == self.players[player-1]:
                return False
        
        # print("Col true")
        return True
        
    def checkDiagonals (self, player):
        status1 = True
        status2 = True
        for i in range (self.size):
            if not self.board[i][i] == self.players[player-1]:
                status1 = False # checking diagonal
        
        r = 0
        c = self.size - 1
        while r < self.size and c >= 0:
            if not self.board[r][c] == self.players[player-1]:
                status2 = False # checking anti-diagonal
            r += 1
            c -= 1
        
        return status1 or status2
```

### Output of Normal Two Player TicTacToe Game
Below is the output of the normal two player tictactoe game. Nothing fancy. Just wanted to check if my implementation works properly. And guess what? It works!!

```python
choose a board size: 
3
Pick your lucky symbol: X or O
X

  |   |  
- + - + -
  |   |  
- + - + -
  |   |  

Let's begin..
Player  1 's turn!
Choose a number betwwen 1 to  9
5

  |   |  
- + - + -
  | X |  
- + - + -
  |   |  

Player  2 's turn!
Choose a number betwwen 1 to  9
1

O |   |  
- + - + -
  | X |  
- + - + -
  |   |  

Player  1 's turn!
Choose a number betwwen 1 to  9
3

O |   | X
- + - + -
  | X |  
- + - + -
  |   |  

Player  2 's turn!
Choose a number betwwen 1 to  9
7

O |   | X
- + - + -
  | X |  
- + - + -
O |   |  

Player  1 's turn!
Choose a number betwwen 1 to  9
4

O |   | X
- + - + -
X | X |  
- + - + -
O |   |  

Player  2 's turn!
Choose a number betwwen 1 to  9
6

O |   | X
- + - + -
X | X | O
- + - + -
O |   |  

Player  1 's turn!
Choose a number betwwen 1 to  9
2

O | X | X
- + - + -
X | X | O
- + - + -
O |   |  

Player  2 's turn!
Choose a number betwwen 1 to  9
8

O | X | X
- + - + -
X | X | O
- + - + -
O | O |  

Player  1 's turn!
Choose a number betwwen 1 to  9
9

O | X | X
- + - + -
X | X | O
- + - + -
O | O | X

===============
One more? (y/n)
===============
n
```

## TicTacToe AI
Extending the basic implementation of TicTacToe and adding necessary features to make it compitable with AI system. 

```python
class TicTacToeAI (TicTacToe):
    
    def __init__(self, size, players):
        super().__init__(size, players)
        
    def allPossibleNextMoves (self):
        possibleMoves = []
        
        for row in range(self.size):
            for col in range(self.size):
                if (self.isValidMove(row, col)):
                    possibleMoves.append([row, col])
        
        return possibleMoves
```

## MiniMax Algorithm
Wiki: Minimax (sometimes MinMax, MM[1] or saddle point[2]) is a decision rule used in artificial intelligence, decision theory, game theory, statistics, and philosophy for minimizing the possible loss for a worst case (maximum loss) scenario. When dealing with gains, it is referred to as "maximin"—to maximize the minimum gain. Originally formulated for n-player zero-sum game theory, covering both the cases where players take alternate moves and those where they make simultaneous moves, it has also been extended to more complex games and to general decision-making in the presence of uncertainty.
Source: https://en.wikipedia.org/wiki/Minimax

### Pseudocode for Wikipedia
Pseudocode of basic minimax algorithm is given below-

```python
function minimax(node, depth, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, minimax(child, depth − 1, FALSE))
        return value
    else (* minimizing player *)
        value := +∞
        for each child of node do
            value := min(value, minimax(child, depth − 1, TRUE))
        return value
```

Basic MiniMax implemtation is given below. Didn't implement Alpha-Beta Pruning.

Thanks to wiki,The Coding Train, levelup and many other online resources to help me understanding and implementing MiniMax: 
https://en.wikipedia.org/wiki/Minimax
https://levelup.gitconnected.com/mastering-tic-tac-toe-with-minimax-algorithm-3394d65fa88f
https://www.youtube.com/watch?v=trKjYdBASyQ&t=1196s

### Find Best Possible Move
Iterate through all the possible moves and call minimax each time to find the best possible move

```python
    def findBestMiniMaxMove (self, player):
        bestScore = -math.inf
        bestMove = None
        counter = [0]
        
        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove[0], possibleMove[1], player)
            score = self.minimax (False, player, 0, counter)
            self.game.undo()

            if (score > bestScore):
                bestScore = score
                bestMove = possibleMove
        
        print ("I have compared ", counter[0], " combinations and executed the best move out of it. I can't lose, dude!")
        return bestMove
```

### Minimax
Return Max Score and Min Score respectively for Maximizing and Minimizing player.

```python
    def minimax (self, isMax, player, depth, counter):
        
        counter[0] = counter[0] + 1
        
        winner = self.game.getWinner()
        if (not (winner == None)):
            if (winner == 0):
                return 0
            elif (winner == player):
                return 10 - depth
            else:
                return depth - 10
        
        maxScore = -math.inf
        minScore = math.inf
        
        for possibleMove in self.game.allPossibleNextMoves():
            currPlayer = player if isMax else 2 if (player == 1) else 1
            
            self.game.move(possibleMove[0], possibleMove[1], currPlayer)
            score = self.minimax (not isMax, player, depth + 1, counter)
            self.game.undo()
            
            if (score > maxScore):
                maxScore = score
            if (score < minScore):
                minScore = score
        
        return maxScore if isMax else minScore
```

## Generating DataSet
We need a good dataset before leveraging machine learning algorithms. There are many ways to generate data. For my model, I generated data after each move. 

### Generate New Row
Add new row to temporary dataset (newData) after every move. 
This is using to generate new data so that we can train our machine learning model with new data after each game.

This function creates two rows for each move considering both of the players as winner 

```python
    def generateNewRow(self):
        newRow = []
        for row in range (self.size):
            for col in range (self.size):
                val = 0
                if (self.board[row][col] == self.players[0]):
                    val = 1
                elif (self.board[row][col] == self.players[1]):
                    val = -1
                
                newRow.append(val)
        
        newInvertRow = [v if v == 0 else -1 if v == 1 else 1 for v in newRow]
        
        self.newData.append (newRow)
        self.newData.append (newInvertRow)
```

### Get New Data
GetNewData will be called at the end of each match. 
This function labels all the data and return a new set of dataset so that we can train our ML model with this new set of data

```python
    def getNewData (self, winner):
        if (winner == 1): 
            newTrainY = [1 if i % 2 == 0 else -1 for i in range(len(self.newData))]
        elif (winner == 2): 
            newTrainY = [-1 if i % 2 == 0 else 1 for i in range(len(self.newData))]
        else: 
            newTrainY = [0 for i in range(len(self.newData))]
        
        newTrainX = self.newData
        self.newData = []
        
        print("Size of newTrainX and newTrainY: ", len(newTrainX), len(newTrainY))
        # print(newTrainX)
        # print(newTrainY)
        
        return newTrainX, newTrainY
```

## Logistic Regression Algorithm
Logistic Regression is well known machine learning algorithm. Not going to detail of this algorithm.

### Find Best Possible Move
Iterate through all the possible moves, generate new test data and call logistic regression algorithm to find the best possible move based on the probability of winning.

```python
    def findBestLogisticMove (self, player):
        testX = []
        positions = []
        
        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove[0], possibleMove[1], player)
            positions.append (possibleMove)
            testX.append (self.generateTestX())
            self.game.undo()
        
        desireClass = 1 if player == 1 else -1
        
        predictions = self.logisticRegressionTesting (testX)
        index = np.where(self.LRModel.classes_ == desireClass)[0][0]
        maxProb = np.amax(predictions[:, index])
        moveIndex = np.where(predictions[:, index] == maxProb)[index][0]
        
        return positions[moveIndex]
```

### Logistic Regression Training
Train your logistic Regression model with the new and old dataset

```python
    def logisticRegressionTraining (self):
        self.LRModel = LogisticRegression(random_state=0).fit(self.trainX, self.trainY)
```

### Logistic Regression Testing
Test your logistic Regression model with all possible moves and find the best move based on the probability of winning.

```python
    def logisticRegressionTesting (self, testX):
        return self.LRModel.predict_proba(testX)
```

### Generate Test Data
Generating testing data based on all possible moves.

```python
    def generateTestX (self):
        newRow = []
        for row in range (self.game.size):
            for col in range (self.game.size):
                val = 0
                if (self.game.board[row][col] == self.game.players[0]):
                    val = 1
                elif (self.game.board[row][col] == self.game.players[1]):
                    val = -1
                
                newRow.append(val)
        return newRow
```

### Battle 1: Human Vs Minimax Algorithm!
Below is the output of Human Vs Minimax!. Quess what? Minimax doesn't lose!

```python
Let's begin..
It's MiniMax's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

O |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's human's turn!
Choose a number betwwen 1 to  9
5

O |   |  
- + - + -
  | X |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

O | O |  
- + - + -
  | X |  
- + - + -
  |   |  

It's human's turn!
Choose a number betwwen 1 to  9
3

O | O | X
- + - + -
  | X |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
  | X |  
- + - + -
O |   |  

It's human's turn!
Choose a number betwwen 1 to  9
4

O | O | X
- + - + -
X | X |  
- + - + -
O |   |  

It's MiniMax's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
X | X | O
- + - + -
O |   |  

It's human's turn!
Choose a number betwwen 1 to  9
8

O | O | X
- + - + -
X | X | O
- + - + -
O | X |  

It's MiniMax's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
X | X | O
- + - + -
O | X | O

Match Draw!
```

### Battle 2: Minimax Vs Minimax!
Below is the output of Minimax Vs Minimax!. No one wins!

```python
Let's begin..

  |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's MiniMax2's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

O |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's MiniMax1's turn!
I have compared  59704  combinations and executed the best move out of it. I can't lose, dude!

O |   |  
- + - + -
  | X |  
- + - + -
  |   |  

It's MiniMax2's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

O | O |  
- + - + -
  | X |  
- + - + -
  |   |  

It's MiniMax1's turn!
I have compared  934  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
  | X |  
- + - + -
  |   |  

It's MiniMax2's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
  | X |  
- + - + -
O |   |  

It's MiniMax1's turn!
I have compared  46  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
X | X |  
- + - + -
O |   |  

It's MiniMax2's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
X | X | O
- + - + -
O |   |  

It's MiniMax1's turn!
I have compared  4  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
X | X | O
- + - + -
O | X |  

It's MiniMax2's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

O | O | X
- + - + -
X | X | O
- + - + -
O | X | O

Match Draw!
```

### Battle 3: Minimax Vs Logistic Regression!
Below is the output of Minimax Vs Logistic Regression!. Minimax is always unbeatable. Logistic Regression is unbeatable as well if it is trained with good dataset. Logistic Regression is way faster than Minimax (obviously minimax would perform faster with alpha-beta prouning.) Logistic Regression can lose first few matches if the model is not trained well, but we are training the model at the end of each match, so logistic regression will definitely make a come back! Fingers Crossed!! 

Data has been gereated by placing some random moves against minimax algorithm.

```python
Training size:  194
======================== Data has been generated =================================
Logictic Regression: Hey! It's learning time. Running towards being a human!
I have learnt lots of new tricks!
======================== IT'S BATTLE TIME =================================
Battle number:  1
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  0
==============================================================
It's MiniMax's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

X |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X |   |  
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

X | X |  
- + - + -
  | O |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
  | O |  
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O |  
- + - + -
X |   |  

It's MiniMax's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O | X
- + - + -
X | O |  

It's MiniMax's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X | O | X

It's over!
Size of newTrainX and newTrainY:  18 18
Training size:  212
Logictic Regression: Hey! It's learning time. Running towards being a human!
I have learnt lots of new tricks!
Match Draw!
Battle number:  2
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  1
==============================================================
It's MiniMax's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

X |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X |   |  
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

X | X |  
- + - + -
  | O |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
  | O |  
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O |  
- + - + -
X |   |  

It's MiniMax's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O | X
- + - + -
X | O |  

It's MiniMax's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X | O | X

It's over!
Size of newTrainX and newTrainY:  18 18
Training size:  230
Logictic Regression: Hey! It's learning time. Running towards being a human!
I have learnt lots of new tricks!
Match Draw!
Battle number:  3
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  2
==============================================================
It's MiniMax's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

X |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X |   |  
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

X | X |  
- + - + -
  | O |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
  | O |  
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O |  
- + - + -
X |   |  

It's MiniMax's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O | X
- + - + -
X | O |  

It's MiniMax's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X | O | X

It's over!
Size of newTrainX and newTrainY:  18 18
Training size:  248
Logictic Regression: Hey! It's learning time. Running towards being a human!
I have learnt lots of new tricks!
Match Draw!
Battle number:  4
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  3
==============================================================
It's MiniMax's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

X |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X |   |  
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

X | X |  
- + - + -
  | O |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
  | O |  
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O |  
- + - + -
X |   |  

It's MiniMax's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O | X
- + - + -
X | O |  

It's MiniMax's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X | O | X

It's over!
Size of newTrainX and newTrainY:  18 18
Training size:  266
Logictic Regression: Hey! It's learning time. Running towards being a human!
I have learnt lots of new tricks!
Match Draw!
Battle number:  5
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  4
==============================================================
It's MiniMax's turn!
I have compared  549945  combinations and executed the best move out of it. I can't lose, dude!

X |   |  
- + - + -
  |   |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X |   |  
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  7331  combinations and executed the best move out of it. I can't lose, dude!

X | X |  
- + - + -
  | O |  
- + - + -
  |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
  | O |  
- + - + -
  |   |  

It's MiniMax's turn!
I have compared  197  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
  | O |  
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O |  
- + - + -
X |   |  

It's MiniMax's turn!
I have compared  13  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X |   |  

It's Logistic Regression's turn!

X | X | O
- + - + -
O | O | X
- + - + -
X | O |  

It's MiniMax's turn!
I have compared  1  combinations and executed the best move out of it. I can't lose, dude!

X | X | O
- + - + -
O | O | X
- + - + -
X | O | X

It's over!
Size of newTrainX and newTrainY:  18 18
Training size:  284
Logictic Regression: Hey! It's learning time. Running towards being a human!
I have learnt lots of new tricks!
Match Draw!
==============================================================
Minimax:  0 Logistic Regression:  0 Draw:  5
==============================================================
```