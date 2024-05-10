import re

# Define patterns and corresponding responses for GitHub-related queries
github_patterns_responses = {
    r'what is github': 'GitHub is a platform for hosting and sharing code repositories. It provides version control using Git, collaboration features, and project management tools.',
    r'how to create a repository': 'To create a repository on GitHub, go to your GitHub account, click on the "+" icon on the top-right corner, and choose "New repository". Follow the instructions to create your repository.',
    r'how to clone a repository': 'To clone a repository from GitHub, use the command "git clone <repository_URL>". Replace <repository_URL> with the URL of the repository you want to clone.',
    r'how to commit changes': 'To commit changes to a repository, use the commands "git add <file_name>" to stage changes and "git commit -m <commit_message>" to commit changes. Replace <file_name> with the name of the file and <commit_message> with your commit message.',
    r'how to push changes': 'To push changes to a GitHub repository, use the command "git push origin <branch_name>". Replace <branch_name> with the name of the branch you want to push to.',
    r'how to pull changes': 'To pull changes from a GitHub repository, use the command "git pull origin <branch_name>". Replace <branch_name> with the name of the branch you want to pull from.',
    r'(.*)': 'I\'m sorry, I don\'t have information about that. Can you please ask something else about GitHub?'
}

# Function to respond to GitHub-related user input
def github_respond(user_input):
    for pattern, response in github_patterns_responses.items():
        if re.match(pattern, user_input.lower()):
            return response
    return "I'm sorry, I don't have information about that. Can you please ask something else about GitHub?"

# Main function to run the chatbot for GitHub
def github_chatbot():
    print("Welcome to the GitHub Chatbot!")
    print("You can ask me anything about GitHub. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye! Have a great day!")
            break
        else:
            print("Bot:", github_respond(user_input))

if __name__ == "__main__":
    github_chatbot()
