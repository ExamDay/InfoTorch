# InfoTorch
### Advanced statistical modeling, analyses, and tests in PyTorch.
With easy hardware acceleration on GPU and TPU.

## Installing
- Clone this repository wherever you want

- Create vitual environment and active it (if you're radical like me and addicted to danger you can skip this step)
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

### skewness_fn(x, dim=1)

- Calculates skewness of data "x" along dimension "dim".

### kurtosis_fn(x, dim=1)

- Calculates kurtosis of data "x" along dimension "dim".

### bimodality_index(x, dim=1)

- Used to detect bimodality (or multimodality) of dataset(s) given a tensor "x" containing the data
  and a dimension "dim" along which to calculate.  The logic behind this index is that a bimodal (or
  multimodal) distribution with light tails will have very low kurtosis, an asymmetric character, or
  both – all of which increase this index.  The smaller this value is the more likely the data are to
  follow a unimodal distribution.  As a rule: if return value ≤ 0.555 (bimodal index for uniform
  distribution), the data are considered to follow a unimodal distribution. Otherwise, they follow a
  bimodal or multimodal distribution.

### KernelDensityEstimate(data: torch.Tensor, x_tics: torch.Tensor = None, start: float = -9, end: float = 9, kernel: torch.distributions.Distribution = Normal(loc=0, scale=1), bandwidth_adjustment: float = 1, dim: int = 1):

- Estimates the probability density function of a batch of data in tensor "data" alond dimension "dim".
- Dimensions after dim are flattened into that dimension with torch.flatten().

### MLE_Fit(model: torch.nn.Module, data: torch.Tensor, dim: int=1, lr: float=5e-2, iters: int=250)

- Fits the parameters of the provided model to the provided data. Provided model must have implimented log_prob() and constrain() methods.

### ECDF(x: torch.Tensor, dim: int = 0)

- Finds empirical cumulative distribution function of provided data "x" along dimension "dim".

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
pdoc --html tools.py --force
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
