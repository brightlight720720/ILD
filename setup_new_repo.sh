#!/bin/bash

# Check if username, token, and repo name are provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 <github_username> <personal_access_token> <new_repo_name>"
  exit 1
fi

# Get the arguments
USERNAME=$1
TOKEN=$2
REPO_NAME=$3

# Set the remote URL with credentials
git remote set-url origin https://$USERNAME:$TOKEN@github.com/$USERNAME/$REPO_NAME.git

# Verify the remote
git remote -v

echo "Remote URL updated to new repository: $USERNAME/$REPO_NAME"
echo ""
echo "Now push your code with:"
echo "git push -u origin main"