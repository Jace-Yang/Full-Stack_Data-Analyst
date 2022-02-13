This is a notebook from Jace Yang.

#### Re-configure the website after making changes on `book`:

```
conda activate /Users/jace/opt/anaconda3/envs/PYFORAML
cd Desktop/GitHub/Full-Stack_Data-Analyst/book
jupyter-book build ./
ghp-import -n -p -f _build/html
rm -r _build
```