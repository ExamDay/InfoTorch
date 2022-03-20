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
That's it!

## Using
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="tools.ECDF"><code class="name flex">
<span>def <span class="ident">ECDF</span></span>(<span>x:&nbsp;torch.Tensor, dim:&nbsp;int&nbsp;=&nbsp;0)</span>
</code></dt>
<dd>
<div class="desc"><p>Finds empirical cumulative distribution function of provided data "x" along dimension "dim".</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python hljs"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">ECDF</span>(<span class="hljs-params">x: torch.Tensor, dim: int = <span class="hljs-number">0</span></span>):</span>
    <span class="hljs-string">'''
    Finds empirical cumulative distribution function of provided data "x" along dimension "dim".
    '''</span>
    x = torch.sort(x.flatten(dim), dim=dim).values
    n = x.shape[<span class="hljs-number">-1</span>]
    cum = torch.arange(<span class="hljs-number">1</span>, n + <span class="hljs-number">1</span>).to(x.device) / n
    cum = cum.repeat(*x.shape[<span class="hljs-number">0</span>:<span class="hljs-number">-1</span>], <span class="hljs-number">1</span>)  <span class="hljs-comment"># one for each univariate sample</span>
    <span class="hljs-keyword">return</span> torch.cat((x.unsqueeze(dim), cum.unsqueeze(dim)), dim)</code></pre>
</details>
</dd>
<dt id="tools.KernelDensityEstimate"><code class="name flex">
<span>def <span class="ident">KernelDensityEstimate</span></span>(<span>data, x_tics=None, start=-9, end=9, kernel=Normal(loc: 0.0, scale: 1.0), bandwidth_adjustment=1, dim=1)</span>
</code></dt>
<dd>
<div class="desc"><p>Estimates the probability density function of a batch of data.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python hljs"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">KernelDensityEstimate</span>(<span class="hljs-params">
    data,
    x_tics=None,
    start=<span class="hljs-number">-9</span>,
    end=<span class="hljs-number">9</span>,
    kernel=Normal(<span class="hljs-params">loc=<span class="hljs-number">0</span>, scale=<span class="hljs-number">1</span></span>),
    bandwidth_adjustment=<span class="hljs-number">1</span>,
    dim=<span class="hljs-number">1</span>,
</span>):</span>
    <span class="hljs-string">'''Estimates the probability density function of a batch of data.'''</span>
    <span class="hljs-comment"># convert to positive index (important for unsqueezing)</span>
    <span class="hljs-keyword">if</span> dim &lt; <span class="hljs-number">0</span>:
        dim = len(data.shape) + dim
        <span class="hljs-keyword">if</span> dim &gt; (len(data.shape) - <span class="hljs-number">1</span>) <span class="hljs-keyword">or</span> dim &lt; <span class="hljs-number">0</span>:
            <span class="hljs-keyword">raise</span> IndexError

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">kde_prob</span>(<span class="hljs-params">
        data,
        x,
        dim=dim,
        kernel=Normal(<span class="hljs-params">loc=<span class="hljs-number">0</span>, scale=<span class="hljs-number">1</span></span>),
        bandwidth_adjustment=<span class="hljs-number">1</span>,
    </span>):</span>
        <span class="hljs-string">'''
        Returns the probability of the items in tensor 'x' according to the PDF estimated by a KDE.
        This function is memory intensive.
        '''</span>
        data = data.flatten(dim)
        n = data.shape[dim]
        silvermans_factor = ((<span class="hljs-number">4</span> * torch.std(data, dim).pow(<span class="hljs-number">5</span>)) / (<span class="hljs-number">3</span> * n)).pow(<span class="hljs-number">1</span> / <span class="hljs-number">5</span>)

        bw = silvermans_factor * bandwidth_adjustment

        bw = bw.view(
            *bw.shape, *[<span class="hljs-number">1</span> <span class="hljs-keyword">for</span> _ <span class="hljs-keyword">in</span> range(len(data.shape) + <span class="hljs-number">1</span> - len(bw.shape))]
        )
        a = data.unsqueeze(dim) - x.unsqueeze(<span class="hljs-number">1</span>)
        a = a / bw
        a = kernel.log_prob(a)
        a = torch.exp(a)
        a = bw ** (<span class="hljs-number">-1</span>) * a
        a = a.sum(dim=dim + <span class="hljs-number">1</span>)
        prob = a / n

        <span class="hljs-keyword">return</span> prob

    <span class="hljs-keyword">if</span> x_tics <span class="hljs-keyword">is</span> <span class="hljs-literal">None</span>:
        <span class="hljs-keyword">assert</span> start <span class="hljs-keyword">and</span> end
        <span class="hljs-keyword">assert</span> end &gt; start
        a = max(torch.min(data).item(), start)  <span class="hljs-comment"># lower integration bound</span>
        b = min(torch.max(data).item(), end)  <span class="hljs-comment"># upper integration bound</span>
        x_tics = torch.Tensor(np.linspace(a, b, steps)).to(data.device)
    <span class="hljs-keyword">else</span>:
        <span class="hljs-keyword">assert</span> isinstance(x_tics, torch.Tensor)
        x_tics = x_tics.to(data.device)
    x_tics.requires_grad = <span class="hljs-literal">True</span>
    kde_y_tics = kde_prob(
        data,
        x_tics,
        kernel=kernel,
        bandwidth_adjustment=bandwidth_adjustment,
    )
    <span class="hljs-keyword">return</span> kde_y_tics</code></pre>
</details>
</dd>
<dt id="tools.MLE_Fit"><code class="name flex">
<span>def <span class="ident">MLE_Fit</span></span>(<span>model, data, dim=1, lr=0.05, iters=250)</span>
</code></dt>
<dd>
<div class="desc"><p>Fits the parameters of the provided model to the provided data. Provided model must have
implimented log_prob() and constrain() methods.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python hljs"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">MLE_Fit</span>(<span class="hljs-params">model, data, dim=<span class="hljs-number">1</span>, lr=<span class="hljs-number">5e-2</span>, iters=<span class="hljs-number">250</span></span>):</span>
    <span class="hljs-string">'''
    Fits the parameters of the provided model to the provided data. Provided model must have
    implimented log_prob() and constrain() methods.
    '''</span>
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    <span class="hljs-comment">#  print("model parameters:", [x for x in model.parameters()])</span>
    <span class="hljs-comment">#  data = data.flatten(dim)</span>
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(iters):
        nll = -torch.sum(model.log_prob(data))
        nll.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.constrain()</code></pre>
</details>
</dd>
<dt id="tools.bimodality_index"><code class="name flex">
<span>def <span class="ident">bimodality_index</span></span>(<span>x, dim=1)</span>
</code></dt>
<dd>
<div class="desc"><p>Used to detect bimodality (or multimodality) of dataset(s) given a tensor "x" containing the
data and a dimension "dim" along which to calculate.
The logic behind this index is that a
bimodal (or multimodal) distribution with light tails will have very low kurtosis, an asymmetric
character, or both – all of which increase this index.
The smaller this value is the more
likely the data are to follow a unimodal distribution.
As a rule: if return value ≤ 0.555
(bimodal index for uniform distribution), the data are considered to follow a unimodal
distribution. Otherwise, they follow a bimodal or multimodal distribution.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python hljs"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">bimodality_index</span>(<span class="hljs-params">x, dim=<span class="hljs-number">1</span></span>):</span>
    <span class="hljs-string">'''
    Used to detect bimodality (or multimodality) of dataset(s) given a tensor "x" containing the
    data and a dimension "dim" along which to calculate.  The logic behind this index is that a
    bimodal (or multimodal) distribution with light tails will have very low kurtosis, an asymmetric
    character, or both – all of which increase this index.  The smaller this value is the more
    likely the data are to follow a unimodal distribution.  As a rule: if return value ≤ 0.555
    (bimodal index for uniform distribution), the data are considered to follow a unimodal
    distribution. Otherwise, they follow a bimodal or multimodal distribution.
    '''</span>
    <span class="hljs-comment"># calculate standard deviation and mean of dataset(s)</span>
    std, mean = torch.std_mean(x, dim)
    <span class="hljs-comment"># get number of samples in dataset(s)</span>
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = <span class="hljs-number">1e-6</span>  <span class="hljs-comment"># for stability</span>

    <span class="hljs-comment"># calculate skewness:</span>
    <span class="hljs-comment"># repeating most of the skewness function here to avoid recomputation of standard devation and mean</span>
    sample_bias_adjustment = torch.sqrt(n * (n - <span class="hljs-number">1</span>)) / (n - <span class="hljs-number">2</span>)
    skew = sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(<span class="hljs-number">3</span>), dim) / n)
        / std.pow(<span class="hljs-number">3</span>).clamp(min=eps)
    )

    <span class="hljs-comment"># calculate kurtosis:</span>
    <span class="hljs-comment"># repeating most the kurtosis function here to avoid recomputation of standard devation and mean</span>
    sample_bias_adjustment = (n - <span class="hljs-number">1</span>) / ((n - <span class="hljs-number">2</span>) * (n - <span class="hljs-number">3</span>))
    kurt = sample_bias_adjustment * (
        (n + <span class="hljs-number">1</span>)
        * (
            (torch.sum((z.T - mean.unsqueeze(dim).T).T.pow(<span class="hljs-number">4</span>), dim) / n)
            / std.pow(<span class="hljs-number">4</span>).clamp(min=eps)
        )
        - <span class="hljs-number">3</span> * (n - <span class="hljs-number">1</span>)
    )

    <span class="hljs-comment"># calculate bimodality index:</span>
    BC = (skew.pow(<span class="hljs-number">2</span>) + <span class="hljs-number">1</span>) / (kurt + <span class="hljs-number">3</span> * ((n - <span class="hljs-number">2</span>).pow(<span class="hljs-number">2</span>) / ((n - <span class="hljs-number">2</span>) * (n - <span class="hljs-number">3</span>))))

    <span class="hljs-keyword">return</span> BC</code></pre>
</details>
</dd>
<dt id="tools.kurtosis_fn"><code class="name flex">
<span>def <span class="ident">kurtosis_fn</span></span>(<span>x, dim=1)</span>
</code></dt>
<dd>
<div class="desc"><p>Calculates kurtosis of data "x" along dimension "dim".</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python hljs"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">kurtosis_fn</span>(<span class="hljs-params">x, dim=<span class="hljs-number">1</span></span>):</span>
    <span class="hljs-string">'''Calculates kurtosis of data "x" along dimension "dim".'''</span>
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = <span class="hljs-number">1e-6</span>  <span class="hljs-comment"># for stability</span>

    sample_bias_adjustment = (n - <span class="hljs-number">1</span>) / ((n - <span class="hljs-number">2</span>) * (n - <span class="hljs-number">3</span>))
    kurtosis = sample_bias_adjustment * (
        (n + <span class="hljs-number">1</span>)
        * (
            (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(<span class="hljs-number">4</span>), dim) / n)
            / std.pow(<span class="hljs-number">4</span>).clamp(min=eps)
        )
        - <span class="hljs-number">3</span> * (n - <span class="hljs-number">1</span>)
    )
    <span class="hljs-keyword">return</span> kurtosis</code></pre>
</details>
</dd>
<dt id="tools.skewness_fn"><code class="name flex">
<span>def <span class="ident">skewness_fn</span></span>(<span>x, dim=1)</span>
</code></dt>
<dd>
<div class="desc"><p>Calculates skewness of data "x" along dimension "dim".</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python hljs"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">skewness_fn</span>(<span class="hljs-params">x, dim=<span class="hljs-number">1</span></span>):</span>
    <span class="hljs-string">'''Calculates skewness of data "x" along dimension "dim".'''</span>
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = <span class="hljs-number">1e-6</span>  <span class="hljs-comment"># for stability</span>

    sample_bias_adjustment = torch.sqrt(n * (n - <span class="hljs-number">1</span>)) / (n - <span class="hljs-number">2</span>)
    skewness = sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(<span class="hljs-number">3</span>), dim) / n)
        / std.pow(<span class="hljs-number">3</span>).clamp(min=eps)
    )
    <span class="hljs-keyword">return</span> skewness</code></pre>
</details>
</dd>
</dl>
</section>

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
