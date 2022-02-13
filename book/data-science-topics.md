# Data Science, what is it?

The image below comes from [Drew Conway's original Venn Diagram blog post](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram). As he states, "I think the term "data science" is a bit of a misnomer," and then he muses on what what is unique about Data Sciene and how it is somewhat distinct from traditional statistics or machine learning. 

```{figure} ./assets/Data_Science_VD.png
[Drew Conway's original Venn Diagram](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram).  
```


There have since been [an enormous number of variations on this diagram](https://www.google.com/search?tbm=isch&as_q=data+science+venn+diagrams&tbs=isz:lt,islt:4mp,sur:fmc), various forms of criticism, etc.; however, it does fit fairly well for this course. This course is primarily aimed at physicists, where it is expected that you will bring your **Substantive Expertise** in physics to bear on problems. 

```{warning}
Notice that the intersection of Hacking Skills and Substantive Expertise (without Math / Statistics Knowledge) is labeled as **Danger Zone!**. 
Much of my goal in this class is to teach you enough statistics that you can avoid the Danger Zone. 
```

Much of the material in this course is aimed at statistics, where I have emphasized first principles and conceptual thinking (eg. transformation properties of the likelihood and posteriors, the interpretation of the prior for parameters that describe fundamental constants of nature, etc.). And you see that much of "traditional research" sits in this overlap. The additonal ingredient that Drew introduces is "Hacking Skills", which is often misunderstood or misinterpreted. "Hacking" here does not refer to computer security or something nefarious -- in fact, quite the opposite. It's probably better to look at something like a [hackathon](https://www.rasmussen.edu/degrees/technology/blog/what-is-a-hackathon/) or [AstroHackWeek](http://astrohackweek.org/2020/), to get a feeling for what is meant. 

```{admonition} Working definition of "Hacking" 
The way that I think of it is the ability to fluently use computing and technology to achieve a goal without being bogged down by those technical details and loosing site of the goal. 
```

```{admonition} Hackathon
“Computer programmers and software designers collaborate and create a solution to an existing problem using technology,” Moore explains. Those participating in a hackathon will work with like-minded individuals to utilize new technologies and hack together tons of code from different sources to achieve the goal, according to Sean Hsieh [source](https://www.rasmussen.edu/degrees/technology/blog/what-is-a-hackathon/),
```

```{admonition} AstroHackWeek
AstroHackWeek is, in part, a summer school. The mornings will offer lectures and exercises covering essential skills for working effectively with large astronomical datasets. Past years have seen topics such as machine learning, Bayesian inference, frequentist statistics, databases, numerical Python, and visualization.
```

You can also see that the intersection of Hacking Skills (think computing) with Math / Statistics Knowledge is labeled as Machine Learning, which I think is reasonable for machine learning in practice (though not for the theory of machine learning). Of course, computationally-minded statisticians often find the terms Machine Learning and Data Science as somewhat offensive, and some people use the term [Computational Statistics](https://en.wikipedia.org/wiki/Computational_statistics). 

Since the diagram is about Data Science, of course it has to be in the middle of the Venn diagram. Importantly, it includes Substantive Expertise. In industry this would mean knowing something about your industry, while in science it means having expertise in your field. For this reason, several people have criticized the "land grab" of statistics and computer science departments trying to claim the term "Data Science" (in an academic context) without [explicitly connecting to domains of practice](http://msdse.org/files/Creating_Institutional_Change.pdf) (eg. physics or astronomy). 


```{admonition} Origin
The term “data science” was first coined in 2008 by [D.J. Patil](https://en.wikipedia.org/wiki/DJ_Patil), and [Jeff Hammerbacher](https://en.wikipedia.org/wiki/Jeff_Hammerbacher), the pioneer leads of data and analytics efforts at LinkedIn and Facebook ([Source](https://aponia.co/development-data-science-ny/)). See also the [interview with D.J. Patil in the Observer](https://observer.com/2019/11/data-scientist-inventor-dj-patil-interview-linkedin-job-market-trend/)
```

Industrial data science often has a lot of overlap with "Big Data", and there is no shortage of figures outlining the essential tools and skills of industrial data science. 
These diagrams can be quite intimidating and should probably be interepreted as the union of the tools and skills that are found and not the intersection of the tools and skills you would need to master.

```{figure} http://nirvacana.com/thoughts/wp-content/uploads/2013/07/RoadToDataScientist1.png
:width: 50%
An example from the myriad of such diagrams mapping the essential tools and skills of industrial data science.  ([Source](https://aponia.co/development-data-science-ny/))
```
```{figure} https://hackr.io/blog/uploads/images/1570190916VwRfvnEiWq.jpg
:width: 50%
Another example from the myriad of such diagrams mapping the essential tools and skills of industrial data science.  ([Source](https://hackr.io/blog/what-is-data-science))
```

```{warning}
This portion of the site is the least well developed, but for now here are some references:
```

## Some References

1. [Veridical Data Science by Bin Yu and Karl Kumbier](https://www.stat.berkeley.edu/~binyu/ps/papers2020/VDS20-YuKumbier.pdf)
1. [other good resources from Rebecca Barter](http://www.rebeccabarter.com/useful_resources/)
1. Scientific Python & Tools
    1. Wonderful jupyterbook by Ryan Abernathey covering many of the topics below [An Introduction to Earth and Environmental Data Science](https://earth-env-data-science.github.io./intro.html)
    1. [Basic Python](https://swcarpentry.github.io/python-novice-inflammation/)
    1. [Plotting and Programming with Python](https://swcarpentry.github.io/python-novice-gapminder/)
    1. numpy
    1. pandas
    1. xarray
    1. sympy
    1. scikit-learn
    1. tensorflow, pytorch, mxnet, jax
    1. Visualization
        1. matplotlib
            1. 3-d plotting [notes](https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html)
        1. seaborn
        1. bokeh
    1. Dask
    1. [Image processing](https://datacarpentry.org/image-processing/)
