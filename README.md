This is a notebook from Jace Yang.

#### Re-configure the website after making changes on `book`:

```
conda activate DL
cd Desktop/GitHub/Full-Stack_Data-Analyst/book
rm -r _build
jupyter-book build ./
ghp-import -n -p -f _build/html
rm -r _build
```