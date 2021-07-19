![image](https://github.com/patternizer/glosat-new-england/blob/master/salem(MA)-holyoke-cambridge(MA)-farrar.png)

# glosat-new-england

Python codebase for construction of a long timeseries of New England temperatures. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `load-new-england-datasets.py` - python script to read in, parse and generate BHO with back-extension 1786-2021 form source data operations
* `load-excel.py` - python helper script to read in BHO observation data from spreadsheets
* `load-ghcnm.py` - python helper script to read in selected station timeseries from GHCNMv4 (QCU and QCF) for processing code

## Instructions for use

The first step is to clone the latest glosat-new-england code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-new-england.git
    $ cd glosat-new-england

Then create a DATA/ directory and copy to it the required inventories listed in python glosat-new-england.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python load-new-england-datasets.py
    $ python load-excel.py
    $ python load-ghcnm.py

This will generate comma separated value (CSV) output files corresponding to the Pandas dataframes constructed from the source data and analysis plots.

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


