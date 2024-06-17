# Git Important Notes

## Authentication

**Step 1: Check Repository URL**

First, ensure that the remote repository URL is correct. You can check this by running:

```
git remote -v
```

If the URL is incorrect, you can update it with:

```
git remote set-url origin <new-url>
```

**Step 2: Check Authentication**

If the repository URL is correct, ensure that your credentials are correctly configured. For repositories hosted on GitHub, you can use a personal access token (PAT) instead of a password.

1. Generate a Personal Access Token (PAT) on GitHub:

- Go to [GitHub's token settings](https://github.com/settings/tokens).
- Click "Generate new token".
- Select the appropriate scopes (usually, *repo* is sufficient).
- Generate the token and copy it.

2. Configure Git to Use the Token:

You can use the token when prompted for a password or save it in your Git configuration.

```
git remote set-url origin https://<your-username>:<your-token>@github.com/RUTILEA/fingerprint-segment-restore.git
```

Replace <your-username> with your GitHub username and <your-token> with the generated PAT.

## Merge (Default Strategy)

**Option 1:No-rebase**

This will create a merge commit to combine the histories.

```
git pull origin develop --no-rebase
```

**Option 2: Rebase**

This will rebase your local commits on top of the remote commits.

```
git pull origin develop --rebase
```

The --rebase option in Git is used to reapply your local commits on top of the upstream changes fetched from the remote repository. This is an alternative to merging, and it can result in a cleaner, linear project history.

Here’s what happens when you use `--rebase`:

- *Fetch the latest changes from the remote branch:* Git first fetches the latest commits from the remote branch you specify.

- *Move your local commits to a temporary area:* Your local commits that are not yet in the remote branch are temporarily removed.

- *Apply the remote commits:* The new commits from the remote branch are applied to your local branch.

- *Reapply your local commits:* Your local commits are then reapplied on top of the newly fetched remote commits.

*Example:* Suppose your commit history looks like this:

```
A---B---C (origin/develop)
     \
      D---E---F (local develop)
```

If you run git pull origin develop --rebase, Git will rebase your local commits D, E, and F on top of the updated origin/develop:

```
A---B---C---D'---E'---F' (local develop)
```

**Option 3: Fast-Forward Only**

This will only update your branch if it can be "fast-forwarded" without creating new commits.

```
git pull origin develop --ff-only
```
