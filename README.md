This is a notebook from Jace Yang.

哈喽，欢迎来到我的笔记库！我是一名哥大DS的学生，在这个笔记的每次commit中，我都在向着全栈DA的理想不断迈进ing，也希望这份笔记可以和自己一起成长，并对你也有帮助！一起加油💪💪


#### Re-configure the website after making changes on `book`｜安装指南:


- Set up｜第一次起笔记环境:
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

- Everytime edit the code｜小幅度更新:

    - Just want to check the result locally｜本地查看:

    ```
    conda activate jupybook
    cd book
    rm -r _build
    jupyter-book build --all ./
    cp -R images _build/html/images
    open -a "Google Chrome" _build/html/index.html
    ```

    - Looks good? Push the result｜网页更新:

    ```
    conda activate jupybook
    cd book
    rm -r _build
    jupyter-book build --all ./
    cp -R images _build/html/images
    open -a "Google Chrome" _build/html/index.html
    ghp-import -n -p -f _build/html
    open -a "Google Chrome" https://github.com/Jace-Yang/Full-Stack_Data-Analyst/deployments/activity_log?environment=github-pages
    ```

- Lots of updates on the pages? Also make the main branch up-to-date｜大幅度更新:

    ```
    rm -r _build
    cd ..
    git add .
    git commit -m "新增因果推断📒"
    git push origin main
    open -a "Google Chrome" https://github.com/Jace-Yang/Full-Stack_Data-Analyst/deployments/
    ```