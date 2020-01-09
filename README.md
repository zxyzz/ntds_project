# Ntds Project 2019

## Inter-County migration and voting pattern at the US election - Team 22

### Pre-requisites
To run the project, you will need python3 along with *ntds_2019* environment. The codes are tested on *Python 3.7.3*.
Dataset can be found at https://drive.google.com/open?id=1Ioyg4-oq0pFppWk9JBIvTj8165BWnAtk (link to a Drive where all the necessary dataset for the project have been saved). 

### Goal of the project
The main goal of this project is to study the inter-county migration in the US for the year 2016 and use the latter study to try to predict the outcome of the Presidential Result at the County level. 

### Datasets
This project uses two datasets regarding the inter-county migration in the US and the result to the 2016 US Presidential Election. 
The inter-county migration data comes from the IRS (US federal agency responsible of the revenues services). A description of the latter dataset is added to the project in the file 1516inpublicmigdoc.pdf or can be found as comments in the file Project.ipynb. The second data comes from the daily newspaper The Guardian. Further details on it can be found also on the notebook. 


### Structure of the codes
This repo contains several files for the entire project. We stongrly encourage the reader to read the Jupyter Notebook Project.ijpnb, for it contains many explainations about the code and the reason that drove certain choice for the projet.  

##### - Project.ipynb  
This is the main file of the project. It is constructed by three following sections:  
 - I. Load, clean, study and prepare the data for graph creation<br/>
   Dataset is loaded and cleaned in the subsection *I.1*. Then a deep study of dataset can be found at the subsection *I.2*. 

 - II. Creation of simple graph following structure of migration & first attempt to predict county type<br/>
   Diverse graphs are created and studied from the dataset to understand the structure of a migration. Graphs are studied through observations using Gephi, degree of county (i.e node of the graph), degree neighboring nodes of county and  ----- Graph observation(Part of anshul?)----
 
 - III. Study of a similarity graph for prediction<br/>
   In this section, we tried to do more sophisticated methods such as GCN and Fourier with Laplacian analyse on different graphs.

##### - utils.py  
This file contains helper functions for graph analyses, including GFT and GCN. Please refer to function doc for further information.

### Authors 
Fatima Moujrid<br/>
Xiaoyan Zou<br/>
Anshul Toshniwal<br/>
Paul Mansat

### License
This project is licensed under the MIT License - see the LICENSE.md file for details

### Resources
Here are the links to the origin files found on the drive:
- Migration data: https://www.irs.gov/statistics/soi-tax-stats-migration-data 
- Election data: https://www.theguardian.com/us-news/ng-interactive/2016/nov/08/us-election-2016-results-live-clinton-trump?view=map&type=presidential 


