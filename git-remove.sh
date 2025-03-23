#!/bin/bash

# Script name: git-remove-and-push.sh
# Purpose: Remove specified files, commit and push these deletion operations through Git
# Usage: ./git-remove-and-push.sh [-m "commit message"] <file_path1> [file_path2 ...]
# Example: ./git-remove-and-push.sh -m "Remove old config files" path/to/file1.txt path/to/directory/*

# Error handling setup
set -e                  # Exit script immediately if any command fails
set -u                  # Error when using undefined variables

# Initialize variables
custom_message=""
files_to_remove=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            if [[ -n "$2" && "$2" != -* ]]; then
                custom_message="$2"
                shift 2
            else
                echo "Error: -m option requires a commit message parameter"
                exit 1
            fi
            ;;
        *)
            files_to_remove+=("$1")
            shift
            ;;
    esac
done

# Check if at least one file parameter was provided
if [ ${#files_to_remove[@]} -eq 0 ]; then
    echo "Error: Please provide at least one file path to remove"
    echo "Usage: $0 [-m \"commit message\"] <file_path1> [file_path2 ...]"
    exit 1
fi

# Check if current directory is a Git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: Current directory is not a Git repository"
    exit 1
fi

removed_files=()

# Prompt user to confirm deletion
echo "Preparing to delete the following files:"
for file in "${files_to_remove[@]}"; do
    echo "  - $file"
done

read -p "Confirm deletion of these files? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Operation cancelled"
    exit 0
fi

# Start removing files
echo "Starting file removal..."

for file in "${files_to_remove[@]}"; do
    # Check if file exists
    if [ ! -e "$file" ]; then
        echo "Warning: File '$file' does not exist, skipping"
        continue
    fi

    # Check if file is tracked by Git
    if ! git ls-files --error-unmatch "$file" &>/dev/null; then
        echo "Warning: File '$file' is not under Git version control, using regular rm to delete"
        rm -rf "$file"
    else
        # First use rm to delete the file
        rm -rf "$file"
        echo "Deleted: $file"

        # Use git rm to record the deletion
        git rm --cached -r "$file" &>/dev/null || true
        echo "Git recorded deletion: $file"

        removed_files+=("$file")
    fi
done

# If files were successfully deleted, commit the changes
if [ ${#removed_files[@]} -gt 0 ]; then
    # Determine commit message
    if [ -n "$custom_message" ]; then
        # Use user-provided custom commit message
        commit_message="$custom_message"
    else
        # Generate default commit message
        if [ ${#removed_files[@]} -eq 1 ]; then
            commit_message="Remove file: ${removed_files[0]}"
        else
            commit_message="Remove multiple files"
            for file in "${removed_files[@]}"; do
                commit_message+="\n- $file"
            done
        fi
    fi

    # Commit changes
    echo "Committing changes..."
    echo "Commit message: $commit_message"
    git commit -m "$commit_message"

    # Push changes
    echo "Pushing changes to remote repository..."
    git push

    echo "Operation complete! Files have been deleted and pushed to the remote repository."
else
    echo "No files were deleted, no commit needed"
fi