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

# Make sure .gitignore contains Replit-specific files
echo "Checking .gitignore for Replit-specific files..."
if ! grep -q "replit.nix" .gitignore; then
  echo "Adding replit.nix to .gitignore"
  echo "replit.nix" >> .gitignore
fi

if ! grep -q ".replit" .gitignore; then
  echo "Adding .replit to .gitignore"
  echo ".replit" >> .gitignore
fi

# Set the remote URL with credentials
git remote set-url origin https://$USERNAME:$TOKEN@github.com/$USERNAME/$REPO_NAME.git

# Verify the remote
git remote -v

echo "Remote URL updated to new repository: $USERNAME/$REPO_NAME"
echo ""
echo "Ready to push your code to GitHub without Replit-specific files!"
echo "Run these commands to push:"
echo "git add ."
echo "git commit -m \"Initial commit for GitHub repository\""
echo "git push -u origin main"