import random

# List of words to guess from
word_list = ['python', 'javascript', 'hangman', 'computer', 'programming', 'game', 'developer']


# Function to display the current state of the word
def display_word(word, guessed_letters):
    return ''.join([letter if letter in guessed_letters else '_' for letter in word])


# Function to give feedback on the guess (green, yellow, gray)
def give_feedback(guess, word):
    feedback = []
    for i in range(len(word)):
        if guess[i] == word[i]:
            feedback.append('🟩')  # Correct letter, correct position (green)
        elif guess[i] in word:
            feedback.append('🟨')  # Correct letter, wrong position (yellow)
        else:
            feedback.append('⬛')  # Incorrect letter (gray)
    return ''.join(feedback)


# Function to play the game
def play_game():
    word = random.choice(word_list)  # Choose a random word from the list
    guessed_letters = []  # List to store guessed letters
    attempts = 6  # Number of incorrect guesses allowed
    guessed_word = False  # Flag to check if the word is guessed correctly

    print("Welcome to Wordle!")
    print("Guess the 5-letter word!")

    # Game loop
    while attempts > 0 and not guessed_word:
        print("\nCurrent word:", display_word(word, guessed_letters))
        print(f"Attempts left: {attempts}")

        guess = input("Guess a 5-letter word: ").lower()  # Take user input and make it lowercase

        # Ensure the guess is a valid 5-letter word
        if len(guess) != 5 or not guess.isalpha():
            print("Please enter a valid 5-letter word.")
            continue

        # Add the guessed letters to the list
        for letter in guess:
            if letter not in guessed_letters:
                guessed_letters.append(letter)

        # Provide feedback on the guess
        feedback = give_feedback(guess, word)
        print("Feedback:", feedback)

        # Check if the word has been completely guessed
        if guess == word:
            guessed_word = True
            print(f"\nCongratulations! You've guessed the word: {word}")
        else:
            attempts -= 1  # Deduct an attempt on incorrect guess

    if not guessed_word:
        print(f"\nGame Over! The word was: {word}")


# Run the game
if __name__ == "__main__":
    play_game()