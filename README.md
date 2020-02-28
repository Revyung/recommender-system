This is a detailed pytorch implementation of collaborative filtering in neural networks benchmarked on the MovieLens dataset. In addition, it will discuss the latest advancement in methodology in particular, how deep learning can be utilized to aid learning user-item interaction function.

Things I have done:
    predict user-item ratings based on collaborative history
    implemented classic matrix factorization in neural network based setting
        adopted dot product of embedding as the function form for ratings, bias unit for users and items, aided with mean item rating vector
        the rmse resulted from simple 100 embedding sizes for user and item ~ 1.18, close to 0.8985 which was the original paper performance using 1000 embedding sizes, so it is believed that performance will improve if that increases
    comparsion to plain SVD method, which has worse performance although it is faster than nn based MF
    shortcomings for generalized matrix factorization:
        1. It cannot incorporate other source of information like change of taste over time
        2. Dot product is the only function to capture user item interations
    implemented deep learning ensemble in addition to matrix factorization machine suggested by reference 2 below
    non-convexity issue of the objective function of NeuMF, gradient-based optimization methods got stucked 
        It is suggested in reference 2 that the initialization plays an important role for the convergence and performance of deep learning models. Since NeuMF is an ensemble of GMF and MLP, they propose to initialize NeuMF using the pretrained models of GMF and MLP. Due to computational resources contraint, optimize two models at once, but to make the convexity problem less prominent, implemented logloss with weighted function to reflect the real loss function value over whole training set instead during batch training, resulted in improved performance of rmse = 1.03 with the same embedding size of 100

References: 
    [Large-scale Parallel Collaborative Filtering for the Netflix Prize](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.173.2797&rep=rep1&type=pdf)
    [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)
