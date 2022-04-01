# InfoTorch
### Advanced statistical modeling, analyses, and tests in PyTorch.
With easy hardware acceleration on GPU and TPU.

## Installing
- Clone this repository wherever you want

- Create a vitual environment and active it (if you're radical like me and addicted to danger you can skip this step)
```bash
cd /path/to/this/repository
python3 -m venv venv
source venv/bin/activate
```
- Install the requirements
```
pip3 install -r requirements.txt
```
- That's it!

## Using

### Functions:
Calculations include:
- skewness
- kurtosis
- bimodality index
- kernel density estimate
- MLE Fit (fit a model by maximum likelihood estimation)
- ECDF (empirical cumulative distribution function)
- Metalog Fit (Closed-form)
- Polynomial Fit (Closed-form, with or without weights)
- Mutual Information

### Classes:
- Normal Model (for example)
- Unbounded Metalog Model

### See our [Documentation](https://www.blackboxlabs.dev/infotorch/documentation) for details.

<!-- ### Functions: -->

<!-- ```python3 -->
<!-- skewness_fn(x, dim=1) -->
<!-- ``` -->

<!-- - Calculates skewness of data "x" along dimension "dim". -->

<!-- ```python3 -->
<!-- kurtosis_fn(x, dim=1) -->
<!-- ``` -->

<!-- - Calculates kurtosis of data "x" along dimension "dim". -->

<!-- ```python3 -->
<!-- bimodality_index(x, dim=1) -->
<!-- ``` -->

<!-- - Used to detect bimodality (or multimodality) of dataset(s) given a tensor "x" containing the data -->
<!--   and a dimension "dim" along which to calculate.  The logic behind this index is that a bimodal (or -->
<!--   multimodal) distribution with light tails will have very low kurtosis, an asymmetric character, or -->
<!--   both – all of which increase this index.  The smaller this value is the more likely the data are to -->
<!--   follow a unimodal distribution.  As a rule: if return value ≤ 0.555 (bimodal index for uniform -->
<!--   distribution), the data are considered to follow a unimodal distribution. Otherwise, they follow a -->
<!--   bimodal or multimodal distribution. -->

<!-- ```python3 -->
<!-- KernelDensityEstimate( -->
<!--     data: torch.Tensor, -->
<!--     x_tics: torch.Tensor = None, -->
<!--     start: float = -9, -->
<!--     end: float = 9, -->
<!--     kernel: torch.distributions.Distribution = Normal(loc=0, scale=1), -->
<!--     bandwidth_adjustment: float = 1, -->
<!--     dim: int = 1 -->
<!-- ) -->
<!-- ``` -->

<!-- - Estimates the probability density function of a batch of data in tensor "data" alond dimension "dim". -->
<!-- - Dimensions after dim are flattened into that dimension with torch.flatten(). -->

<!-- ```python3 -->
<!-- MLE_Fit(model: torch.nn.Module, data: torch.Tensor, dim: int=1, lr: float=5e-2, iters: int=250) -->
<!-- ``` -->

<!-- - Fits the parameters of the provided model to the provided data. Provided model must have implimented log_prob() and constrain() methods, and paraters set to some initial value. -->

<!-- ```python3 -->
<!-- ECDF(x: torch.Tensor, dim: int = 0) -->
<!-- ``` -->

<!-- - Finds empirical cumulative distribution function of provided data "x" along dimension "dim". -->

<!-- ### Classes: -->
<!-- ```python3 -->
<!-- Normal_Model( -->
<!--     init_mean: torch.Tensor = torch.Tensor([0]), -->
<!--     init_std: torch.Tensor = torch.Tensor([1]), -->
<!-- ) -->
<!-- ``` -->
<!-- - Example of a module for modeling a probability distribution. This is set up with all pieces required for use with the rest of this package. (initial parameters; as well as implimented constrain, forward, and log_prob methods) -->

## ToDo:

- Mutual information estimation.
- Interaction information estimation.

## Contributing
For contributors to the project; do this before making your first commit:

- Install pre-commit
```bash
cd /path/to/this/repository/
sudo apt install pre-commit
pre-commit install
```
(we do all of our development on linux for now)

- Make sure to update the documentation to include your changes before commiting:
```bash
pdoc --html infotorch.py --force
```

- Also Make sure to include any new dependencies in the requirements.txt before commiting with:
```bash
pip3 freeze > requirements.txt
```

- To test updates to the readme and other GitHub flavored markdown, simply install Grip
and feed it your desired file.
```bash
pip3 install grip
python3 -m grip README.md
```

- Then follow the link provided by the Grip sever for a live preview of your work.

- When satisfied with your changes you can compile to an html file with:
```bash
python3 -m grip README.md --export README.html
```


## Authors
* **Gabe M. LaFond** - *Initial work* - [ExamDay](https://github.com/ExamDay)

See also the list of [contributors](https://github.com/ExamDay/InfoTorch/contributors) who participated in this project.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
