This is a notebook from Jace Yang.

#### Re-configure the website after making changes on `book`:


- Set up:
```
conda activate DL
cd Desktop/GitHub/Full-Stack_Data-Analyst/book
```

- Everytime edit the code and want to see result:

```
rm -r _build
jupyter-book build --all ./
cp -R images _build/html/images
open -a "Google Chrome" _build/html/index.html
```

- Looks good? Push the result:

```
cd book
rm -r _build
jupyter-book build --all ./
cp -R images _build/html/images
ghp-import -n -p -f _build/html
rm -r _build

cd ..
git add .
git commit -m "新增因果推断模块——统计章节——基础——辛普森悖论📒"
git push origin main
open -a "Google Chrome" https://github.com/Jace-Yang/Full-Stack_Data-Analyst/deployments/
```