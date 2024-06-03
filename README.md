# Master thesis ☘️ 🏭🦾
Repository for my masters thesis in *Chemical engineering* .


| Title                 | **Modeling of CO2 Capture using Machine Learning**                                 |
|-----------------------|------------------------------------------------------------------------------------|
| Subtitle              | **A data-driven approach for specific reboiler duty prediction of a pilot plant** |
| Author                | **Jun Xing Li**                                                                   |
| Supervisor            | **Hanna Katariina Knuutila**                                                      |
| Co-supervisor         | **Idelfonso Bessa dos Reis Nogueira, Ricardo Ramos Wanderley (Aker Carbon Capture)** |
| Date                  | **June 2024**                                                                     |


The full version of my thesis can be found on NTNU Open. [INSERT LINK]

# Abstract
As nations worldwide strive to meet emission reduction goals set by the Paris Agreement, carbon capture and storage plays a crucial role in mitigating greenhouse gas emissions. The full-scale carbon capture and storage project based on CO2 emission from industrial sources *Longship* puts Norway at the forefront of technology development, with Aker Carbon Capture as the technology provider. The operation of carbon capture plants requires significant amounts of energy, and accurate models of plants are essential for energy optimization and reducing operating costs. Data-driven modeling techniques, particularly machine learning, have gained traction in recent years for various applications in the carbon capture domain. However, most studies have relied on first-principle models or data from small-scale pilot plants, which may not accurately represent real-world scenarios.

To further advance the field of data-driven modeling in carbon capture, it is essential to explore the use of more extensive datasets acquired from large-scale pilot plants. This thesis aims to develop machine learning models based on data from Aker Carbon Capture's mobile test unit pilot plant, focusing on predicting the specific reboiler duty of the desorber. The objectives include identifying necessary preprocessing steps, employing a domain knowledge-based approach for determining system stability, and exploring the strengths and weaknesses of different machine learning models for interpolation and extrapolation. The project will develop two categories of models: interpolation-based models using traditional linear models, ensemble decision trees, and artificial neural networks, and extrapolation models focusing on deep learning approaches. The dataset will be partitioned into steady-state and transient-state subsets to assess model performance under different dynamics. 

The results demonstrate the potential of machine learning for predicting specific reboiler duty, with interpolation models achieving high accuracy using random forest and artificial neural networks. Extrapolation models, particularly those based on long short-term memory networks, show promise for forecasting, especially during steady-state operation. Both types of models yield less accurate models during transient states and underline the importance of handling predictions outside of normal operating conditions. The findings highlight the importance of data quantity, preprocessing, and model architecture in achieving accurate predictions.

Future work will focus on multi-step time series forecasting, more advanced preprocessing methods for handling high values and expanding the dataset to include various solvents and flue gas compositions. This study contributes to the progress of data-driven modeling in carbon capture, laying the groundwork for more efficient energy optimization, ultimately aiding in reducing greenhouse gas emissions and fighting climate change.


## Table of Contents
* [Disclaimer](#disclaimer)
* [Requirements](#requirements)
* [Description](#description)
* [Deeper dive into the notebooks](#deeper-dive-into-the-notebooks)
* [Nice resources](#nice-resources)



## Disclaimer
This repository is meant as an archive where I can store my code. I hope that even suboptimal code can serve as inspiration for students who also want to explore machine learning. There are many potential improvements in my code, so please be critical and don't assume everything is correct 😉. I hope you find something useful here, and if you have any questions, please don't hesitate to contact me.

<p> <a href="https://www.linkedin.com/in/lijunxing/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" alt="linkedin"> LinkedIn </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://github.com/Xingern" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/tskMh.png" alt="github"> Github </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://junxingli.no" rel="nofollow noreferrer"> 👻 My website </a> </p>


## Requirements
Run the following command to install all of the packages listed in `environment.yml`. This assumes you have conda installed; if not, install it first. I recommend using  [MiniConda](https://docs.anaconda.com/free/miniconda/).

```bash
conda env create -f environment.yml
```


## Description
The code provided contains everything I wrote in Python that contributed to my thesis. This includes all the final models, figures, source code, and data. The LaTeX code for my report is also uploaded so that one can reconstruct parts of the layout or use it as a template.

1. `data/`: The folder where the data is stored, sorted by either `raw` or `processed`. The code assumes that the data is stored here.

2. `figs/`: The location for all of the figures generated across the notebooks. The figures are sorted by type of model and the model name. The folders containing `Case X` is a macOS shortcut folder, not sure how it works on other computers. 

3. `models/`: The stored location of all of my models, except the linear models (even though RF is non-linear) since that are easy to train. 

4. `notebooks/`: The backbone of my thesis and all code is ran from these notebooks. The folder named legacy is some unused code that I dont want to delete. 

5. `scripts/`: The most commonly used functions that I moved outisde of the notebooks. Especially the plotting functions are redundant and therefore defined across all notebooks. 


## Deeper dive into the notebooks

1. `ANN-SFS.ipynb`: I moved the whole SFS method into a separate notebook since it contains a smaller search space than ordinary ANN. However, ANN-SFS is not something I would recommend since it takes a really looong time to run. I might be crazy to do that, but it worked since my dataset is relative small. 

2. `ANN-TimeSeries.ipynb`: Adjusted the input layer to become a matrix and using an autoregressive approach to ANNs. 

3. `ANN.ipynb`: The ordinary notebook for ANNs which contains hyperparameter tuning using `Hyperband()`.

4. `generate_plots.ipynb`: Notebook for generating various plots such as the final results, but also the some for the theory section. There are examples of PCA on palmerpenguins and UMAP on MNIST.

5. `LM.iypnb`: A collection of LR-SFS, EN and RF methods into a large class. Built to easily handle an input dataset, filter based on stability. Was written in OOP to refresh those skills and not entirely necesseary. 

6. `LSTM.ipynb`: The most complex model, but is acutally very similar to ANN, you just import another layer. Don't be scared of the implementation, its basically the same as in `ANN.ipynb`.

7. `preprocessing.ipynb`: The most important preprocessing steps organized into a large notebook. Take in a raw dataset and turn it into processed data. All of the steps are described in detail in my thesis. The order of the preprocessing steps slightly differs, however the content is the same.

8. `legacy/`: Some code from earlier stages during my masters thesis and I did not bother to remove. There may be something nice that can be used one day, or not. Anyway, its not something important so ignore this folder.




## Nice resources

I personally took these 3 courses at NTNU, which form the foundation of my machine learning knowledge. However, don't feel discouraged if you're starting from scratch - motivation is everything!

### Courses at NTNU

I have personally taken these 3 courses at NTNU which is the foundation of my machine learning knowledge. However, do not feel discouraged if you are starting from scratch, motivation is everything!

* **TMA4268 - Statistisk læring**: This was my first course that introduced me to statistical learning (which is mostly the same as machine leanring). Great course even if you only know basic statistics, however it can be pretty math-hevay sometimes.
* **TKJ4175 - Kjemometri**: A really nice course that introduces some machine learning concepts, especially PCA. Pretty nice introduction and a relative light course to take.
* **TTK4260 - Multivariat dataanalyse og maskinlæring**: My favorite course since it describes a wider range of methods. However there are no compulsory exercises so it's easy to underestimate this course. 


### Videos

The YouTube videoes are absolutely GOATED so there are especially two creators I want to mention: 3Blue1Brown and StatQuest. 

### Books

There are tons of books out there in machine learning and that field. I want to highlight 3 books that I have used and are really nice. 

* *[An Introduction to Statistical Learning](https://www.statlearning.com)*: The greatest book on machine learning from a statistical view. The course book in TM4268 and the concepts are explained very detailed and easiy to understand. Some of the authors even invented the methods described in the book (LASSO). 
* *[Machine Learning with PyTorch and Scikit-Learn](https://www.akademika.no/teknologi/data-og-informasjonsteknologi/machine-learning-pytorch-and-scikit-learn/9781801819312)*:The most updated book in terms of advanced techniques such as graph neural networks, transformers and natural language processing. The book explains many complex concepts in a easy manner with many notebooks in their repository. 
* *[Deep Learning](https://www.deeplearningbook.org)*: A famous book by *Goodfellow et al.* that many machine learning people swear to. Goodfellow developed what is today called *Generative adversarial network* aka. generative AI.  

Two of the above books are free and easily accessable from the web. The last book can also be found using more shady site or just buy it from your local provider (Akademika have it).

