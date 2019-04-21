# AI Games
### Part 1: Pichu
###### Part of Prof D.J. Crandall's B-551/2017 Assignment 2
<p>
Chess has been called the "drosophila of artificial intelligence," since it has long been a convenient yardstick
by which to measure progress in AI (just as the fruit fly has been a relatively simple experimental "platform"
for biology). Let's consider Pichu, a somewhat simplified version of Chess that is popular among a certain
community of midwestern bird enthusiasts.

The game is played by two players on a board consisting of a grid of 8 x 8 squares. Initially, each player
has sixteen pieces: 8 Parakeets, 2 Robins, 2 Nighthawks, 2 Blue jays, 1 Quetzal, and 1 Kingfisher. The two
players alternate turns, with White going first. On each turn, a player moves exactly one of his or her pieces,
possibly capturing (removing) a piece of the opposite player in the process, according to the following rules:
  • A Parakeet may move one square forward, if no other piece is on that square. Or, a Parakeet may
    move one square forward diagonally (one square forward and one square left or right) if a piece of the
    opposite player is on that square, in the process capturing that piece from the board. If a Parakeet
    reaches the far row of the board (closest to the opposite player), it is transformed into a Quetzal. On
    its very first move of the game, a Parakeet may move forward two squares as long as both are empty.
  • A Robin may move any number of squares either horizontally or vertically, landing on either an empty
    square or a piece of the opposite player (which is then captured), as long as all the squares between
    the starting and ending positions are empty.
  • A Blue jay is like a Robin, but moves along diagonal 
    ight paths instead of horizontal or vertical ones.
  • A Quetzal is like a combination of a Robin and a Blue jay: it may move any number of empty squares
    horizontally, vertically, or diagonally, and land either on an empty square or on a piece of the opposite
    player (which is then captured).
  • A Kingfisher may move one square in any direction, horizontally or vertically, either to an empty square
    or to capture a piece of the opposing player.
  • A Nighthawk moves in L shaped patterns on the board, either two squares to the left or right followed
    by one square forward or backward, or one square left or right followed by two squares forward or
    backward. It may fly over any pieces on the way, but the destination square must either be empty or
    have a piece of the opposite player (which is then captured).

A player wins by capturing the other player's Kingfisher. (Note some of the differences with traditional
Chess: there's no notion of check or checkmate, no en passant, and no castling.)

Your task is to write a Python program that plays Pichu well. Use the minimax algorithm with alpha-beta
search and a suitable heuristic evaluation function. Your program should accept a command line argument
that gives the current state of the board as a string of 64 characters, each of which is one of: . for an empty
square, P or p for a white or black Parakeet, R or r for a white or black Robin, N or n for a white or black
Nighthawk, Q or q for a white or black Quetzal, K or k for a white or black Kingfisher, and B or b for a white
or black Blue jay, in row-major order. For example, the encoding of the start state of the game would be:
          RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr

More precisely, your program will be called with three command line parameters: (1) the current player (w or
b), (2) the state of the board, encoded as above, and (3) a time limit in seconds. Your program should then
decide a recommended single move for the given player from the given current board state, and display the
new state of the board after making that move, within the number of seconds specied. Displaying multiple
lines of output is fine as long as the last line has the recommended board state. (This is an easy way of
dealing with the time limit: the program can very quickly calculate and print a suggested \rough-draft"
move, and then print out better moves as it finds them; our test programs will kill your program after the
time limit has passed and look only at the last move.) For example, a sample run of your program might
look like:

[djcran@macbook]$ ./pichu.py w RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr 10
Thinking! Please wait...
Hmm, I'd recommend moving the Parakeet at row 2 column 3 to row 4 column 3.
New board:
RNBQKBNRPP.PPPPP..........P.....................pppppppprnbqkbnr


In your source code comments, explain your heuristic function and how you arrived at it.
The tournament. To make things more interesting, we will hold a competition among all submitted solutions.
We will not reveal ahead of time the time limit, but we plan to hold multiple tournaments with different
values. While the majority of your grade will be on correctness, programming style, etc., a small portion may
be based on how well your code performs in the tournaments, with particularly well-performing programs
eligible for prizes including extra credit points.
Note: Your code must conform with the interface standards mentioned above! The last line of the output
must be the new board in the format given, without any extra characters or empty lines. Also, note that
your program cannot assume that the game will be run in sequence from start to end; given a current board
position on the command line, your code must find a recommended next best move. Your program can write
files to disk to preserve state between runs, but should correctly handle the case when a new board state is
presented to your program that is unrelated to the last state it saw.

```
### Part 2: Tweet classifcation
```
A classic application of Bayes Law is in document classification. Let's examine one particular classication
problem: estimating where a Twitter \tweet" was sent, based only on the content of the tweet itself. We'll
use a bag-of-words model, which means that we'll represent a tweet in terms of just an unordered \bag" of
words instead of modeling anything about its grammatical structure. In other words, a tweet can be modeled
as simply a histogram over the words of the English language (or, more generally, all possible tokens that
occur on Twitter). If, for example, there are 100,000 words in the English language, then a tweet can be
represented as a 100,000-dimensional binary vector, where in each dimension there is a 1 if the word appears
in the tweet and a zero otherwise. Of course, vectors will be very sparse (most entries are zero).

Implement a Naive Bayes classifier for this problem. For a given tweet D, we'll need to evaluate P(L =
ljw1;w2; :::;wn), the posterior probability that a tweet was taken at one particular location (e.g., l = Chicago)
given the words in that tweet. Make the Naive Bayes assumption, which says that for any i 6= j, wi is
independent from wj given L.

To help you get started, we've provided a dataset in your GitHub repo of tweets, labeled with their actual
geographic locations, split into a training set and a testing set. We've restricted to a set of a dozen North
American cities (Chicago, Philadelphia, etc.), so your task is to classify each tweet into one of twelve dierent
categories. Train your model on the training data and measure performance on the testing data in terms of
accuracy (percentage of documents correctly classified).

Your program should accept command line arguments like this:
 
 ./geolocate.py training-file testing-file output-file

The program should then load in the training file, estimate the needed probabilities to build a Bayesian
model, and apply them to each tweet in the testing file, and then write the results into output-file. The file
format of the training and testing files is simple: one tweet per line, with the rst word of the line indicating
the actual location. Output-file should have the same format, except that the rst word of each line should
be your estimated label, the second word should be the actual label, and the rest of the line should be the
tweet itself. Your program should also output (to the screen) the top 5 words associated with each of the 12 locations (i.e. the words for which P(L = ljw) is the highest for each l).

The goal is to get as high an accuracy as possible in testing, including on the separate test dataset we'll use
to test your code. You'll have to make various design decisions in doing this, e.g. whether to use all \words"
(i.e. Twitter tokens) or just the most common ones, whether to keep punctuation or remove it, whether to
keep capitalization or remove it, etc. Please describe these design decisions and any experimentation you
use to arrive at them in your report.

Hints: Don't worry, at least at first, about whether the \words" in your model are actually words. Just
treat every unique space-delimited token you encounter as a \word," even if it's misspelled, a number, a
punctuation mark, etc. It may be helpful to ignore tokens that do not occur more than a handful of times,
however. To perform classification, you'll need to compute the posterior probability for each of the 12 cities
and then choose the maximum. Note that this means you don't have to actually compute the prior on words,
i.e. the denominator of Bayes Law, since it is the same across all 12 categories and is always positive (so
that maximizing the numerator is the same as maximizing the full posterior).

Extra credit
(Don't attempt unless you're completely happy with your submission for the rest of the assignment. This
problem is mostly for fun. We want you to learn how to write clear, concise, easy-to-understand, easy-
to-maintain code, whereas this problem encourages writing code that is hard to maintain and difficult to
understand. But still, it may be fun :-).

In a momentary lapse in common sense, during a recent lecture an otherwise mild-mannered professor of
computer science made the bold claim that he \could implement a program to play perfect Chess in 20 lines
of Python code."1 The point wasn't to brag about his coding abilities, but simply to underscore the point
that in theory, game-playing AI can be easily implemented; the hard part is implementing it in a way that
avoids simply searching the space of all possible games. Unfortunately, several persistent students called
his bluff, leading to some frantic late-night, jet lag-assisted coding to prove to himself he could do it. His
(theoretically) perfect chess program takes 17 lines of code, and a more practical version with a heuristic
function takes 19. Can you do better? The rules are: (1) you're not allowed to import any modules, (2) blank
lines do not count, but multi-line statements count for multiple lines, (3) must use reasonably meaningful
variable and function names, (4) no line may have more than 225 characters, (5) I/O, test code, etc. doesn't
count, (6) assume the simplified Pichu rules above, and (7) no external data or program files may be loaded
or used.

<p>
