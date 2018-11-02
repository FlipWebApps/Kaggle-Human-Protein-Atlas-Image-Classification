# Kaggle Kaggle Human Protein Atlas Image Classification

Code for the [Kaggle Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)

## Usage
1. Install git and checkout the [git code repository]
2. Install [anaconda] python version 3.6+
3. Change working directory into the git code repository root
4. Create the self contained conda environment. In a terminal go to the git code repository root and enter the command:

   `conda env create --file conda_env.yml`

5. Any python modules under src need to be available to other scripts. This can be done in a couple of ways. You can 
setup and install the python modules by executing the setup.py command below which will install the packages to the 
conda environments site-packages folder but with a symlink to the src folder so modifications are reflected immediately. 

   `python setup.py develop`
   
    As an alternative you may prefer to set the python path directly from the console, within notebooks, test scripts 
    etc. From Pycharm you can also right click the src folder and select the _Mark Directory As | Source Root_ option.

6. Download data from the [Kaggle page](https://www.kaggle.com/c/human-protein-atlas-image-classification/data) and place 
inside the data/raw folder. Extract train.zip and test.zip into new folders train and test. You can either do this
manually or using the [Kaggle api](https://github.com/Kaggle/kaggle-api) (setup credentials as documented)

   ```
   cd data/raw
   kaggle competitions download -c human-protein-atlas-image-classification
   unzip train.zip -d train
   unzip test.zip -d test
   rm train.zip
   rm test.zip
   ```

7. Models include:
    * notebooks/models/U-Net.ipynb - notebook using U-Net architecture
    * scripts/crf.py - Conditional Random Fields model. Run this on any submissions file to generate updated output. 
    
8. Submissions can be uploaded manually through the web page or using the 
[Kaggle api](https://github.com/Kaggle/kaggle-api) (setup credentials as documented)

    ```
    kaggle competitions submit human-protein-atlas-image-classification -f submission.csv.7z -m "My submission message"
    ```
    
NOTE: When working in the project notebooks from within the Equinor network, you may need to include the lines below if your proxy is not otherwise setup.

`os.environ['HTTP_PROXY']="http://www-proxy.statoil.no:80"`<br />
`os.environ['HTTPS_PROXY']="http://www-proxy.statoil.no:80"`

## Using the Python Conda environment

Once the Python Conda environment has been set up, you can

* Activate the environment using the following command in a terminal window:

  * Windows: `activate my_environment`
  * Linux, OS X: `source activate my_environment`
  * The __environment is activated per terminal session__, so you must activate it every time you open terminal.

* Deactivate the environment using the following command in a terminal window:

  * Windows: `deactivate my_environment`
  * Linux, OS X: `source deactivate my_environment`
               
* Delete the environment using the command (can't be undone):

  * `conda remove --name my_environment --all`

## Initial File Structure

```
├── .gitignore               <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if 
│                               needed
├── conda_env.yml            <- Conda environment definition for ensuring consistent setup across environments
├── README.md                <- The top-level README for developers using this project.
├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
│                               generated with `pip freeze > requirements.txt`. Might not be needed if using conda.
├── setup.py                 <- Metadata about your project for easy distribution.
│
├── data
│   ├── processed            <- The final, canonical data sets for modeling.
│   ├── raw                  <- The original, immutable data dump.
│   └── temp                 <- Temporary files.
│
├── notebooks                <- Notebooks for analysis and testing
│   ├── eda                  <- Notebooks for EDA
│   │   └── example.ipynb    <- Example python notebook
│   ├── modelling            <- Notebooks for modelling
│   └── preprocessing        <- Notebooks for Preprocessing 
│
├── src                      <- Code for use in this project.
│   └── examplepackage       <- Example python package - place shared code in such a package
│       ├── __init__.py      <- Python package initialisation
│       ├── examplemodule.py <- Example module with functions and naming / commenting best practices
│       ├── features.py      <- Feature engineering functionality
│       ├── io.py            <- IO functionality
│       └── pipeline.py      <- Pipeline functionality
│
└── tests                    <- Test cases (named after module)
    ├── test_notebook.py     <- Example testing that Jupyter notebooks run without errors
    ├── examplepackage       <- examplepackage tests
        ├── examplemodule    <- examplemodule tests (1 file per method tested)
        ├── features         <- features tests
        ├── io               <- io tests
        └── pipeline         <- pipeline tests
```

## Testing
Reproducability and the correct functioning of code are essential to avoid wasted time. If a code block is copied more 
than once then it should be placed into a common script / module under src and unit tests added. The same applies for 
any other non trivial code to ensure the correct functioning.

To run tests, install pytest using pip or conda (should have been setup already if you used the conda_env.yml file) and 
then from the repository root run
 
```pytest```

## Important Links
* https://wiki.statoil.no/wiki/index.php/Statoil_Data_Science_Technical_Standards - Data Science Technical Standards (Statoil Internal)
* https://dataplatformwiki.azurewebsites.net/doku.php - Data Platform wiki (Statoil internal)
* https://github.com/Statoil/data-science-shared - Shared Data Science Code Repository (Statoil internal)

## References
* https://github.com/Statoil/data-science-template/ - The master template for this project
* http://docs.python-guide.org/en/latest/writing/structure/
* https://github.com/Azure/Microsoft-TDSP
* https://drivendata.github.io/cookiecutter-data-science/

[//]: #
   [anaconda]: <https://www.continuum.io/downloads>
