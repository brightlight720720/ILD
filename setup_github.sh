#!/bin/bash

# Check if username and token are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <github_username> <personal_access_token>"
  exit 1
fi

# Get the username and token from arguments
USERNAME=$1
TOKEN=$2

# Set the remote URL with credentials
git remote set-url origin https://$USERNAME:$TOKEN@github.com/brightlight720720/ILD.git

# Verify the remote
git remote -v

echo "Remote URL updated successfully. You can now run: git push origin main"