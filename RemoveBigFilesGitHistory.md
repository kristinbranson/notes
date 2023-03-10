# Purging big files from git history

We had some large binary files in our git history that were making our repo large. The first few tools I found for removing these files from history didn't seem to work for me, but that could be user error. [git-filter-repo](https://github.com/newren/git-filter-repo) worked.

Steps:
1. [Install](https://github.com/newren/git-filter-repo/blob/main/INSTALL.md). I downloaded the python script [`git-filter-repo`](https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo). 
2. Make a clean, bare clone of the repo with the argument `--use-local`:
```
git clone --mirror --no-local badgit/ badgit_clean.git
```
3. Change into this directory:
```
cd badgit_clean.git
```
4. Find big files, following [these instructions](https://stackoverflow.com/questions/10622179/how-to-find-identify-large-commits-in-git-history):
```
grep -vF --file=<(git ls-tree -r HEAD | awk '{print $3}') | git rev-list --objects --all |   git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |   sed -n 's/^blob //p' |   sort --numeric-sort --key=2 |   cut -c 1-12,41- |   $(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest
```
5. Here are some commands to:
a. Remove any file with the name `badbasename`
```
python3 git-filter-repo --use-base-name --path badbasename --invert-paths
```
b. Remove a directory `my/baddir`
```
python3 git-filter-repo --path-match my/baddir --invert-paths
```
c. Remove a file `my/badfile`
```
python3 git-filter-repo --path-match my/badfile --invert-paths
```
d. Remove all files bigger than 10M:
```
python3 git-filter-repo --strip-blobs-bigger-than 10M
```

