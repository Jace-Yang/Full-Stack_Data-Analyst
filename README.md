This is a notebook from Jace Yang.

#### Re-configure the website after making changes on `book`:


- Set up:
```
conda create --name jupybook
conda activate jupybook
conda install -c conda-forge python=3
conda install -c conda-forge 'jupyterlab>=3.0.0,<4.0.0a0' jupyterlab-lsp
pip install 'python-lsp-server[all]'
pip install -U jupyter-book
pip install ghp-import
pip install sphinx-inline-tabs
pip install sphinx-proof
```

- Everytime edit the code and want to see result:

```
conda activate jupybook
cd book
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
open -a "Google Chrome" _build/html/index.html
ghp-import -n -p -f _build/html
rm -r _build

cd ..
git add .
git commit -m "新增数仓📒"
git push origin main
open -a "Google Chrome" https://github.com/Jace-Yang/Full-Stack_Data-Analyst/deployments/
```