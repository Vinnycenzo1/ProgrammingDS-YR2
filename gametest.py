import random

# List of words to guess from
word_list = ['python', 'javascript', 'hangman', 'computer', 'programming', 'game', 'developer']


# Function to display the current state of the word
def display_word(word, guessed_letters):
    return ''.join([letter if letter in guessed_letters else '_' for letter in word])


# Function to play the game
def play_game():
    word = random.choice(word_list)
    guessed_letters = []
    attempts = 6  # Number of incorrect guesses allowed
    guessed_word = False

    print("Welcome to the Word Guessing Game!")

    # Game loop
    while attempts > 0 and not guessed_word:
        print("\nCurrent word:", display_word(word, guessed_letters))
        print(f"Attempts left: {attempts}")

        guess = input("Guess a letter: ").lower()

        # Check if the input is a single letter
        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single letter.")
            continue

        if guess in guessed_letters:
            print("You've already guessed that letter!")
            continue

        # Add the guessed letter to the list
        guessed_letters.append(guess)

        # Check if the letter is in the word
        if guess in word:
            print(f"Good guess! '{guess}' is in the word.")
        else:
            print(f"Sorry! '{guess}' is not in the word.")
            attempts -= 1

        # Check if the word has been completely guessed
        if all(letter in guessed_letters for letter in word):
            guessed_word = True

    # End of game
    if guessed_word:
        print(f"\nCongratulations! You've guessed the word: {word}")
    else:
        print(f"\nGame Over! The word was: {word}")


# Run the game
if __name__ == "__main__":
    play_game()
How
it
works:
You’ll
guess
one
letter
at
a
time.
The
program
will
show
you
the
word
with underscores for the letters you haven't guessed.
Each
wrong
guess
reduces
the
number
of
attempts
left.
If
you
guess
the
word
correctly or run
out
of
attempts, the
game
ends.
Running
the
Game:
If
you
want
to
run
this, you
can
copy
this
code
into
a
Python
environment(like
IDLE, or any
text
editor
with Python installed) and run it.The game will run in the terminal or command line, and you'll interact with it by typing your guesses.

Let
me
know if you
want
to
add
features or modify
the
game in any
way!



You
said:
when
i
run
it
i
get
a
modulenotfounderror
ChatGPT
said:
Ah, I
see! The
random
module is part
of
Python’s
standard
library, so
it
should
be
available
by
default.The
ModuleNotFoundError
usually
means
that
either:

There’s
an
issue
with your Python installation.
There
might
be
a
typo in the
import statement.

Let
's troubleshoot:

Step
1: Double - check
the
import statement

Ensure
that
you
have
this
line
at
the
top
of
your
code:

python
Copy
Edit
import random

This
should
not cause
any
issues if you
're using a standard Python environment.

Step
2: Verify
your
Python
installation
Check
Python
Version: Run
the
following in your
terminal
to
make
sure
Python is installed
correctly:

bash
Copy
Edit
python - -version
or if you're using Python 3 specifically:

bash
Copy
Edit
python3 - -version
This
should
return the
version
of
Python
you
're using.

Test
the
random
module: Open
a
Python
shell(just
type python or python3 in your
terminal) and run
this:

python
Copy
Edit
import random

print(random.choice([1, 2, 3, 4, 5]))
This
should
print
a
random
number
from

1
to
5
without
any
errors.If
you
get
an
error
here, there
might
be
an
issue
with your Python installation.

Step
3: Environment
Issues
If
you
're using an IDE or code editor (like PyCharm, VS Code, etc.), make sure that your IDE is using the correct Python interpreter. Sometimes, a specific environment might not have all the necessary libraries or could be misconfigured.

If
you’re
still
seeing
errors
after
these
checks, let
me
know
the
exact
error
message
you
're getting, and I can help further!

You
said:
Re - print
the
original
code
but
make
it
work
on
pycharm
ChatGPT
said:
Got
it! Below is the
original
code
with some small adjustments to ensure it should work smoothly in PyCharm or any standard Python environment.

Please
make
sure
you
're running this in a Python file (e.g., word_game.py) in PyCharm and that your Python interpreter is set up correctly.

Here
's the updated code:

python
Copy
Edit
import random

# List of words to guess from
word_list = ['python', 'javascript', 'hangman', 'computer', 'programming', 'game', 'developer']


# Function to display the current state of the word
def display_word(word, guessed_letters):
    return ''.join([letter if letter in guessed_letters else '_' for letter in word])


# Function to play the game
def play_game():
    word = random.choice(word_list)  # Choose a random word from the list
    guessed_letters = []  # List to store guessed letters
    attempts = 6  # Number of incorrect guesses allowed
    guessed_word = False  # Flag to check if the word is guessed correctly

    print("Welcome to the Word Guessing Game!")

    # Game loop
    while attempts > 0 and not guessed_word:
        print("\nCurrent word:", display_word(word, guessed_letters))
        print(f"Attempts left: {attempts}")

        guess = input("Guess a letter: ").lower()  # Take user input and make it lowercase

        # Check if the input is a single letter
        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single letter.")
            continue

        if guess in guessed_letters:
            print("You've already guessed that letter!")
            continue

        # Add the guessed letter to the list of guessed letters
        guessed_letters.append(guess)

        # Check if the letter is in the word
        if guess in word:
            print(f"Good guess! '{guess}' is in the word.")
        else:
            print(f"Sorry! '{guess}' is not in the word.")
            attempts -= 1  # Deduct an attempt on incorrect guess

        # Check if the word has been completely guessed
        if all(letter in guessed_letters for letter in word):
            guessed_word = True

    # End of game
    if guessed_word:
        print(f"\nCongratulations! You've guessed the word: {word}")
    else:
        print(f"\nGame Over! The word was: {word}")


# Run the game
if __name__ == "__main__":
    play_game()