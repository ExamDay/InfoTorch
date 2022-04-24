#! bin/bash
pdoc --html infotorch.py --force
cp html/infotorch.html ~/Desktop/blackbox-rnd-ether/blackboxlabs/main/templates/main/infotorch/documentation.html
cd ~/Desktop/blackbox-rnd-ether
git add blackboxlabs/main/templates/main/infotorch/documentation.html
git commit -m "documentation update" -n
git push
