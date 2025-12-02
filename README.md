# Somnium

Nice dream project

## Generate GitHub contributions

This repository contains a small script that writes to `data.json` and commits it across a range of past dates so you can simulate multiple contributions on GitHub. Use this carefully and only on a repository where this behavior is intended.

Usage:

- Default (260 commits):

```bash
npm start
```

- Custom count (e.g., 250):

```bash
node index.js 250
```

- Dry-run (don't push, just create local commits):

```bash
npm run start:dry
```

Note: Keep your environment configured for git and be sure you have permission to push to the remote repository if you choose to push.

Important note on dry-run:

- `--dry-run` prevents the script from pushing commits to the remote. However, it still creates commits locally; if you don’t want local commits, run this on a throwaway branch or revert afterwards.

Example: Create 100 commits in 2024 (dry-run):

```bash
node index.js 100 2024 --dry-run
```