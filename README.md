This is a notebook from Jace Yang.

#### Re-configure the website after making changes on `book`:


- Set UP:
```
conda activate DL
cd Desktop/GitHub/Full-Stack_Data-Analyst/book
```

- Everytime edit the code:

git add .
cd Desktop/GitHub/Full-Stack_Data-Analyst/book
rm -r _build
jupyter-book build --all ./
cp -R images _build/html/images
ghp-import -n -p -f _build/html
rm -r _build
```