# InfoTorch
### Facilitates hardware acceleration (GPU, TPU) of advanced statistical modeling, analyses, and tests in PyTorch.

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
[TBD]

## Contributing
For contributors to the project; do this before making your first commit:

- Install pre-commit
```bash
cd /path/to/this/repository/
sudo apt install pre-commit
pre-commit install
```
(we do all of our development on linux for now)

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
