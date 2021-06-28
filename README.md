# ear-analytics

A tool to automatically read and visualize results provided by files get by 
_eacct_ command from [EAR](https://gitlab.bsc.es/ear_team/ear/-/wikis/home) framework.
**ear-analytics** is a cli program written in Python which lets you plot the data given by 
the output get when executing `eacct -j %slurm_job_id% -c %file_name%` or 
`eacct -j %slurm_job_id% -l -c %file_name%`. For more information, read about
[eacct](https://gitlab.bsc.es/ear_team/ear/-/wikis/Commands#energy-account-eacct) 
command.

## Requirements

Python 3.6.x

Output of `pip freeze` on a virtual environment used while deploying the 
application:
```
cycler==0.10.0
kiwisolver==1.3.1
matplotlib==3.3.4
numpy==1.19.5
pandas==1.1.5
Pillow==8.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2021.1
six==1.16.0
```

## Usage

