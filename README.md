# 0nset
Data analysis to find the onset value

While the present repository provides source code for a python
implementation, actual usage is directed towards a javascript 
implementation freely available at [dyesdb.com](https://dyesdb.com/onset).

The code used for "Accurate Determination of the Onset
Wavelength (onset) in Optical Spectroscopy" is found in
v_2.0.7

The updated 0nset v_2.1.1 contains the same algorithm in
v_2.0.7 but has fewer issues with user input. Additionally, 
some useful local minima and maxima information is provided 
with each dataset to aide the user in selecting better 
intervals.

# Usage
While the initial implementation of 0nset used a flask application in python, a
much more user-friendly version is available at
[https://dyesdb.com/onset](https://dyesdb.com/onset). This version is written
in javascript and does not require user setup. If you have issues running your
code through either implementation, please don't hesitate to contact me at
[awallace43@gatech.edu](mailto:awallace43@gatech.edu)!

# Setup

## Installation
There are several methods for setting up your python environment 
to run this code; however, one of the easiest methods will be described
below through using Anaconda.

1. Go to [Anaconda](https://www.anaconda.com/products/individual) 
   and download the individual version for your operating system.
2. Either clone the repo with git or download the zip file for this repository.
3. Open the terminal (linux/macOS) or powershell (Windows 10/11) and navigate to 
   where you have extracted the zip file. 
4. Run the following command in the terminal to create your virtual environment.
```bash
conda env create -f environment.yml
```
5. Activate the environment.
```bash
conda activate 0nset
```
6. Designate the flask app
```bash
export FLASK_APP=app.py
```
7. Run the flask app
```bash
flask run
```
8. Open your [localhost](http://127.0.0.1:5000/)
     in the browser.
9. Upload your .csv file (ensure x-axis data increments by 1 for this version) 
10. Follow the procedure outlined in this [paper](http://dx.doi.org/10.1016/j.jqsrt.2021.107544)

