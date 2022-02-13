# Draft Schedule


Recording of lectures are accessible [here](https://applications.zoom.us/lti/rich?lti_scid=53e5b6ade1d092bbd38974bc31813aa0b9c9d37154d82fb28fd97562e49a6c2c&oauth_consumer_key=egAB3MeoRVG9kMt4z8eEXA). 

1. Week 1
    1. 9/2: Intro [Recording](https://nyu.zoom.us/rec/play/uA6Jy6FEsbZpQRfmeQ6jJV7ISs37lvdKMnwsJhuql9O445ANIB0TbmflpCqFPjqczKuhAr3k9voEv8Tc.K2AoCHSjTknMuoAA)
        1. syllabus
        1. juypter book
        1. review survey 
        1. about me and my research and a preview for the course
2. Week 2 
    1. 9/9: Basic prob theory [Recording](https://nyu.zoom.us/rec/play/Y6tMRTUxNDhU2n_pw9cPJr1kqmZDMcfCuqLGRqWuLIQ42M9AhGtuhjS-vi_XfaNzNr2i-nFlBPrSJdRa.eh480MdOBuleGYe2)
        1. Random Variables
            1. Probability space
        1. Probability Mass and Density functions
        1. Conditional Probability
        1. Bayes Theorem
            1. Quantifying prior odds via betting
            1. Incoherent beliefs
        1. Axioms of probability
        1. Examples
3. Week 3 
    1. 9/14: Class [Recording](https://nyu.zoom.us/rec/play/6Wvz4mMAK3qWkBLhOP98kLQMNcEc-H6rZg9zWUao-DTzme6TBAblX8Q7d_0Imyzjts9o48IAOG6reJst.KSmnHtGH6BTNiCvb)
        1. Conditional probability for continuous variables
            1. Chain rule of probability
            1. Sneak peek at graphical models
            1. The Drake equation
            1. Phosphine on Venus and Bayes Theorem
        1. Marginal Distributions
        1. Independence
        1. Emperical Distribution
        1. Expectation
        1. Variance, Covariance, Correlation
        1. Mutual Information
        1. Simple Data Exploration
    1. 9/16: Class [Recording](https://nyu.zoom.us/rec/play/ryKzr2yN2nWSWtMzavZivnrJDZ7rQFoowx5Pk6mWdFKq5ESJFjk0zCGQEtk6G1qCDM2VvdDez6t5Tdzk.OSHZVTqbRrQ-i_wa)
        1. Likelihood
        1. Change of variables
        1. Demo change of variables with autodiff
        1. Independence and correlation
        1. Conditioning
        1. Autoregressive Expansion
        1. Graphical Models
4. Week 4
    1. 9/21: [Recording](https://nyu.zoom.us/rec/play/uSMzP3UYoZBRnDjAQfdzDKC5_WHAmX_tenfl7jduYPoTRqAXfuBYyC-tALiVJEWNNNYChZ-BwDoxe2lz.JdLbnvKxO5vZmXDv)
        1. Change of variables formula
        1. Probability Integral Transform
        1. Intro to automatic differentiation
            1. Demo with automatic differentiation
        1. Transformation properties of the likelihood
        1. Transformation properties of the MLE
        1. Transformation properties of the prior and posterior
        1. Transformation properties of the MAP
    1. 9/23: Estimators
        1. Skipped material from last lecture
            1. Lorentz-invariant phase space
            1. Normalizing Flows
            1. Copula
        1. Bias, Variance, and Mean Squared Error
        1. Simple Examples: Poisson and Gaussian
        1. Cramer-Rao bound & Information Matrix
        1. Bias-Variance tradeoff 
            1. James-Stein Demo
            1. Shrinkage
        1. HW: 
            1. James Stein
5. Week 5
    1. 9/28 (Yom Kippur): Random Numbers [Recording](https://nyu.zoom.us/rec/play/L-BkDSdARQfotstBjZjW8WzTzF2g35bvftQIXWVe5MEmYDyJzscjqs3qrwDrAjKKV8lgHi04hw6EjyLZ.6DP4VVZA6_LvfkWE)
        1. Decision Theory
            1. [Admissible decision rule](https://en.wikipedia.org/wiki/Admissible_decision_rule)
            1. generalized decision rules ("for some prior")
        1. Consistency
        1. Sufficiency
        1. Exponential Family
        1. Score Statistic
        1. Information Matrix
            1. Information Geometry
            1. Transformation properties of Information Matrix
            1. Jeffreys' prior
                1. Transformation properties
            1. Reference Prior
            1. Sensitivity analysis
            1. likelihood principle
    1. 9/30: Lecture 8: Consistency and homework
        1. [Neyman Scott phenomena](https://www.stat.berkeley.edu/~census/neyscpar.pdf) (an example of inconsistent MLE)
            1. Note: [Elizabeth Scott](https://en.wikipedia.org/wiki/Elizabeth_Scott_(mathematician)) was an astronomer by background. In 1957 Scott noted a bias in the observation of galaxy clusters. She noticed that for an observer to find a very distant cluster, it must contain brighter-than-normal galaxies and must also contain a large number of galaxies. She proposed a correction formula to adjust for (what came to be known as) the Scott effect. 
            1. Note: [Revisiting the Neyman-Scott model: an Inconsistent MLE or an Ill-defined Model?](https://arxiv.org/abs/1301.6278)
        1. walk through of nbgrader and home work assignment
6. Week 6
    1. 10/5: Lecture 9: Propagaion of Errors
        1. a simple example from physics 1: estimating $g$
        1. Change of variables vs. Error propagation
        1. Demo Error propagation fails
        1. Error propagation and Marginalization 
        1. Convolution
        1. Central Limit Theorem
        1. Error propagation with correlation
            1. track example
    1. 10/7: Lecture 10: Likelihood-based modeling 
        1. Building a probabilistic model for simple physics 1 example
        1. Connection of MLE to traditional algebraic estimator
        1. Connection to least squares regression
7. Week 7
    1. 10/12 Lecture 11: Sampling
        1. Motiving examples:
            1. Estimating high dimensional integrals and expectations
            1. Bayesian credible intervals
            1. Marginals are trivial with samples
        1. Generating Random numbers
            1. Scipy distributions
        1. Probability Integral Transform
        1. Accept-Reject MC
            1. Acceptance and efficiency
            1. native python loops vs. numpy broadcasting        
        1. Importance Sampling & Unweighting
            1. [Vegas](https://en.wikipedia.org/wiki/VEGAS_algorithm)
        1. Connetion to Bayesian Credible Intervals
        1. Metropolis Hastings MCMC
            1. Proposal functions 
        1. Hamiltonian Monte Carlo
            1. Excerpts from [A Conceptual Introduction to Hamiltonian Monte Carlo by Michael Betancourt](https://arxiv.org/abs/1701.02434)
            1. Stan and PyMC3
    1. 10/14: Lecture 12: Hypothesis Testing and Confidence Intervals
        1. Simple vs. Compound hypotheses
        1. TypeI and TypeII error
        1. critical / acceptance region
        1. Neyman-Pearson Lemma
        1. Test statistics
        1. Confidence Intervals
        1. Interpretation
        1. Coverage
        1. Power
        1. No UMPU Tests
        1. Neyman-Construction
        1. Likelihood-Ratio tests
        1. Connection to binary classification
            1. prior and domain shift
8. Week 8
    1. 10/19: Lecture 13: 
        1. Simple vs. Compound hypotheses
            1. Nuisance Parameters
        1. Profile likelihood
        1. Profile construction
        1. Pivotal quantity
        1. Asymptotic Properties of Likelihood Ratio
            1. Wilks 
            1. Wald
    1. 10/21 Canceled
9. Week 9
    1. 10/26: Lecture 14 
        1. Upper Limits, Lower Limits, Central Limits, Discovery
        1. Power, Expected Limits, Bands
        1. Sensitivity Problem for uppper limits
            1. CLs
            1. power-constrained limits        
    1. 10/28: Lecture 15 flip-flopping, multiple testing
        1. flip flopping
        1. multiple testing
            1. look elsewhere effect
            1. Familywise error rate 
            1. False Discovery Rate
            1. Hypothesis testing when nuisance parameter is present only  under the alternative
                1. [Asymptotics, Daves, Gross and Vitells](https://arxiv.org/abs/1005.1891)
10. Week 10
    1. 11/2 Lecture 16 Combinations, probabilistic modelling languages, probabilistic programming
        1. Combinations
            1. Combining p-values
            1. combining posteriors
            1. likelihood-based combinations
            1. likelihood publishing
        1. probabilistic modelling languages
            1. computational graphs
        1. Probabilistic Programming
            1. First order PPLs
                1. Stan
            1. Universal Probabilistic Programming
                1. pyro
                1. pyprob and ppx
                1. Inference compilation
    1. 11/4 Lecture 17: Goodness of fit
        1. conceptual framing
        1. difference to hypothesis testing
        1. chi-square test
        1. Kolmogorov-Smirnov
        1. Anderson-Darling
        1. Zhang's tests
        1. Bayesian Information Criteria
        1. software
        1. anomaly detection    
11. Week 11
    1. 11/9: Lecture 18 Intro to machine learning
        1. Supervised Learning
        1. Statistical Learning Theory
            1. Loss, Risk, Emperical Risk
            1. Generalization
            1. VC dimension and Emperical risk minimization
            1. No Free Lunch
        1. Cross-validation test/train
            1. Preview: the mystery of deep learning
        1. Least Squares
        1. Regularized least squares
        1. Bayesian Curve fitting
        1. Bias-Variance tradeoff
    1. 11/11 Lecture 19
        1. Generalization
        1. Loss functions for regression 
        1. loss function for classification
        1. Information theory background
            1. Entropy
            1. Mutual information
            1. cross entropy
            1. Relative Entropy
12. Week 12
    1. 11/16: Lecture 20 Density Estimation, Deep Generative Models
        1. Unsupervised learning
        1. Loss functions for density estimation
            1. Divergences
                1. KL Divergence
                1. Fisher distance
                1. Optimal Transport
                1. Hellinger distance
                1. f-divergences
                1. Stein divergence
        1. Maximum likelihood (Forward KL)
            1. can approximate with samples, don't need target distribution
        1. Variational Inference (Reverse KL)
            1. Connecton to statistical physics
            1. LDA (Topic Modelling)
            1. BBVI 
        1. Deep Generative models
            1. Normalizing Flows intro
            1. background on auto-encoders
            1. Variational Auto-encoder intro

    1. 11/18: Lecture 21 Deep Generative Models 
        1. Deep Generative models comparison
            1. Normalizing Flows
            1. Autoregresive models
            1. Variational Auto-encoder
            1. GANs
13. Week 13
    1. 11/23: Lecture 22 The data manifold
        1. what is it, why is it there
            1. in real data
            1. in GANs etc.
        1. How it complicates distances based on likelihood ratios
        1. Optimal transport
    1. 11/25 Lecture 23 Optimization
        1. Gradient descent
        1. Momentum, Adam
        1. Differences of likelihood fits in classical statistics and loss landscape of deep learning models
        1. stochastic gradient descent and mini-batching intro
            1. what is it
14. Week 14
    1. 11/30: Lecture 23 Stochastic gradient descent
        1. Robbins-Monro
        1. connection to Langevin dynamics and approximate Bayesian inference
    1. 12/2: Lecture 24 Implicit bias and regularization in learning algorithms
        1. dynamics of gradient descent
        1. Double descent
15. Week 15
    1. 12/7 Lecture 25 Deep Learning
        1. Loss landscape
            1. random matrix theory
            1. connection to statistical mechanics
        1. Deep Model Zoo
            1. MLP
            1. Convolutions
            1. Sequence Models: RNN and Tree RNN
                1. vanishing and exploding gradients
            1. Graph Networks
            1. Transformers
            1. images, sets, sequences, graphs, hyper-graphs
            1. DL and functional programming
            1. Differentiable programming
    1. 12/9: Review
        1. Review 




## Other topics that we touched on or planned to touch on. 

I need to move some of these topics that we discussed into the schedule.
This is a place holder for now.

1. examples
    1. unbinned likelihood exponential example
1. HW ideas
    1. Conditional Distribuutions
    1. Bernouli to Binomial
    1. Binomial to Poisson
    1. Poisson to Gaussian
    1. Product of Poissons vs. Multinomial
    1. CLT to Extreme Value Theory
    1. [Neyman Scott Phenomena](https://blog.richmond.edu/physicsbunn/2016/11/28/the-neyman-scott-paradox/)
    1. some other shrinkage?
    1. Jeffreys for examples
    1. prior odds via betting example
    1. [Negatively biased relevant subsets](https://arxiv.org/abs/1109.2023)
    1. Group Project: interactive Neyman-Construction Demo
1. Simulation-based inference
    1. ABC
    1. Diggle
    1. likleihood ratio
    1. likelihood
    1. posterior
    1. Mining Gold
1. Topics to Reschedule    
    1. Parametric vs. non-parametric
    1. Non-parametric
        1. Histograms
            1. Binomial / Poisson statistical uncertainty
            1. weighted entries
        1. Kernel Density Estimation
            1. bandwidth and boundaries
            1. K-D Trees
    1. Parameterized
        1. Unsupervised learning
        1. Maximum likelihood
            1. loss function
            1. Neural Denstiy Estimation
        1. Adversarial Training
            1. GANs
            1. WGAN        
    1. Latent Variable Models
    1. Simulators
    1. Connections
        1. graphical models
        1. probability spaces
        1. Change of variables
    1. GANs
1. Classification 
    1. Binary vs. Multi-class classification
    1. Loss functions
    1. logistic regression
    1. Softmax
    1. Neural Networks
    1. Domain Adaptation and Algorithmic Fairness
1. Kernel Machines and Gaussian Processes
    1. Warm up with N-Dim Gaussian
    1. Theory
    1. Examples
1. Causal Inference
    1. ladder of causality
    1. simple examples
    1. Domain shift, inductive bias
    1. Statistical Invariance, pivotal quantities, Causal invariance
    1. [Elements of Causal Inference by  Jonas Peters, Dominik Janzing and Bernhard Schölkopf](https://mitpress.mit.edu/books/elements-causal-inference) [free PDF](https://www.dropbox.com/s/dl/gkmsow492w3oolt/11283.pdf)