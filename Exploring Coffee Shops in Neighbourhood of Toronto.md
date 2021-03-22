
# Exploring Coffee Shops in Neighborhood of Toronto (IBM Data Science Capstone)

**By:- Komal Thakkar**

__**Introduction & Business Purpose:**__

As a part of the IBM Data Science Professional Certification Program, I worked on these datasets to get an real experience of data science. Main objectives of this project were to define a business problem, look for data in the web and, use Foursquare location data to compare different neighborhoods of Toronto to figure out the number of coffee shops in neighbourhood of Toronto & thereby decide which neighborhood is suits best for competitive market.

The purpose for this notebook is to explore the neighborhood of Toronto and generate statistics and analytics on top of it. The introductory part has data cleansing & preprocessing, followed by Analysis of most widely spread coffee shop in neighbourhood of Toronto.

![Inside_the_Coffee_Shop,_Parliament_House,_Dolgellau_-_geograph.org.uk_-_1708041.jpg](attachment:Inside_the_Coffee_Shop,_Parliament_House,_Dolgellau_-_geograph.org.uk_-_1708041.jpg)

**Problem Description :**

Assuming that a person wants to setup a new Coffee Shop in neighbourhood of Toronto. And the person wants to explore all competitors around the city of Toronto and who has how many outlets in the neighbourhood, to grasp the hold of particular vendor. So to overcome doubts in his mind whether it is a good idea to open a shop or not, Data Analysis and report is required. In order to make the business profitable, significant analysis before investing is must.

**Area of Benefit:**

There can be significant benficiaries through analysis of this project:

• Business Person who wants to open an outlet of coffee in the neighbourhood.

• The end-users i.e. ardent coffee fans.

• Enterprise of Data Analyst / Data Scientist who analyses the neighbourhood using statistical Data.



**Data Acquisition**

Below are the list of sources from which I have collected the data for different purpose:

1. Postal Codes for Canada :-
· I have fetched the postal code of the neighborhoods in Canada by scraping Wikipedia.
· Link — https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=1011037969


2. Geographical Co-ordinates :-
· I used a CVS file which consist latitude and longitude of the neighborhoods in Canada.
· I chose the CVS file instead using geocoder.
· Link for CVS — http://cocl.us/Geospatial_data


3. Foursqaure for Fetching Details of the venue :
· I used Foursquare API for fetching the details and location of the venues.
· And finally visualize using Folium.


From Foursquare API (https://developer.foursquare.com/docs),

I retrieved the following for each venue:
a) Name: The name of the venue.
b) Category: The category type as defined by the API.
c) Latitude: The latitude value of the venue.
d) Longitude: The longitude value of the venue.
e) Likes: Likes of the venue, that the user liked the restaurant.
f) Rating: Rating of the venue.
g) Tips: Tips given by the users.



**Installing necessary libraries:**


```python
!pip install --upgrade numpy
```

    Requirement already satisfied: numpy in /opt/conda/envs/Python-3.7-main/lib/python3.7/site-packages (1.18.5)
    Collecting numpy
      Downloading numpy-1.20.1-cp37-cp37m-manylinux2010_x86_64.whl (15.3 MB)
    [K     |████████████████████████████████| 15.3 MB 16.2 MB/s eta 0:00:01
    [?25hInstalling collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.18.5
        Uninstalling numpy-1.18.5:
          Successfully uninstalled numpy-1.18.5
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    ibm-watson-machine-learning 1.0.53 requires pandas<=1.0.5, but you have pandas 1.2.3 which is incompatible.[0m
    Successfully installed numpy-1.20.1


The project includes scraping & cleansing of the Wikipedia page for codes of Canada and then processing the data for the clustering.

**Scraping the Wikipedia Source page for the table of postal codes of Canada**

**Step-1:Installing and Importing the required Libraries**


```python
!pip install beautifulsoup4
!pip install lxml
import requests # library to handle requests
import pandas as pd # library for data analsysis
#import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

!conda install -c conda-forge geopy --aayes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 


from IPython.display import display_html
import pandas as pd
import numpy as np
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors

print('Folium installed')
print('Libraries imported.')
```

    Requirement already satisfied: beautifulsoup4 in /opt/conda/envs/Python-3.7-main/lib/python3.7/site-packages (4.9.3)
    Requirement already satisfied: soupsieve>1.2 in /opt/conda/envs/Python-3.7-main/lib/python3.7/site-packages (from beautifulsoup4) (2.0.1)
    Requirement already satisfied: lxml in /opt/conda/envs/Python-3.7-main/lib/python3.7/site-packages (4.6.2)
    usage: conda [-h] [-V] command ...
    conda: error: unrecognized arguments: --aayes
    Collecting package metadata (current_repodata.json): done
    Solving environment: | 
    The environment is inconsistent, please check the package plan carefully
    The following packages are causing the inconsistency:
    
      - conda-forge/linux-64::pytorch==1.8.0=cpu_py37hafa7651_0
      - defaults/noarch::ibm-wsrt-py37main-keep==0.0.0=1962
      - defaults/noarch::ibm-wsrt-py37main-main==custom=1962
    done
    
    # All requested packages already installed.
    
    Folium installed
    Libraries imported.



```python
!pip install Jupyter_to_medium

```

    Requirement already satisfied: Jupyter_to_medium in /Users/itskt/anaconda3/lib/python3.7/site-packages (0.2.4)
    Requirement already satisfied: nbconvert in /Users/itskt/anaconda3/lib/python3.7/site-packages (from Jupyter_to_medium) (5.4.0)
    Requirement already satisfied: matplotlib>=3.1 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from Jupyter_to_medium) (3.3.4)
    Requirement already satisfied: numpy in /Users/itskt/anaconda3/lib/python3.7/site-packages (from Jupyter_to_medium) (1.15.4)
    Requirement already satisfied: requests in /Users/itskt/anaconda3/lib/python3.7/site-packages (from Jupyter_to_medium) (2.21.0)
    Requirement already satisfied: beautifulsoup4 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from Jupyter_to_medium) (4.6.3)
    Requirement already satisfied: mistune>=0.8.1 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (0.8.4)
    Requirement already satisfied: jinja2 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (2.10)
    Requirement already satisfied: pygments in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (2.3.1)
    Requirement already satisfied: traitlets>=4.2 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (4.3.2)
    Requirement already satisfied: jupyter_core in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (4.4.0)
    Requirement already satisfied: nbformat>=4.4 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (4.4.0)
    Requirement already satisfied: entrypoints>=0.2.2 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (0.2.3)
    Requirement already satisfied: bleach in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (3.0.2)
    Requirement already satisfied: pandocfilters>=1.4.1 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (1.4.2)
    Requirement already satisfied: testpath in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (0.4.2)
    Requirement already satisfied: defusedxml in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbconvert->Jupyter_to_medium) (0.5.0)
    Requirement already satisfied: cycler>=0.10 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.1->Jupyter_to_medium) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.1->Jupyter_to_medium) (2.3.0)
    Requirement already satisfied: python-dateutil>=2.1 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.1->Jupyter_to_medium) (2.7.5)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.1->Jupyter_to_medium) (1.0.1)
    Requirement already satisfied: pillow>=6.2.0 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.1->Jupyter_to_medium) (8.1.2)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from requests->Jupyter_to_medium) (2020.12.5)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from requests->Jupyter_to_medium) (1.24.1)
    Requirement already satisfied: idna<2.9,>=2.5 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from requests->Jupyter_to_medium) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from requests->Jupyter_to_medium) (3.0.4)
    Requirement already satisfied: MarkupSafe>=0.23 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from jinja2->nbconvert->Jupyter_to_medium) (1.1.0)
    Requirement already satisfied: six in /Users/itskt/anaconda3/lib/python3.7/site-packages (from traitlets>=4.2->nbconvert->Jupyter_to_medium) (1.12.0)
    Requirement already satisfied: decorator in /Users/itskt/anaconda3/lib/python3.7/site-packages (from traitlets>=4.2->nbconvert->Jupyter_to_medium) (4.3.0)
    Requirement already satisfied: ipython-genutils in /Users/itskt/anaconda3/lib/python3.7/site-packages (from traitlets>=4.2->nbconvert->Jupyter_to_medium) (0.2.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /Users/itskt/anaconda3/lib/python3.7/site-packages (from nbformat>=4.4->nbconvert->Jupyter_to_medium) (2.6.0)
    Requirement already satisfied: webencodings in /Users/itskt/anaconda3/lib/python3.7/site-packages (from bleach->nbconvert->Jupyter_to_medium) (0.5.1)
    Requirement already satisfied: setuptools in /Users/itskt/anaconda3/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib>=3.1->Jupyter_to_medium) (40.6.3)


**Step-2: Using BeautifulSoup Library of Python, for web scraping of table from the Wikipedia:**


```python
source = requests.get('https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=1011037969').text
soup=BeautifulSoup(source,'lxml')
print(soup.title)                         #Printing title of the webpage to check if the page has been scraped 
from IPython.display import display_html
tab = str(soup.table)
display_html(tab,raw=True)
```

    <title>List of postal codes of Canada: M - Wikipedia</title>



<table class="wikitable sortable">
<tbody><tr>
<th>Postal Code
</th>
<th>Borough
</th>
<th>Neighbourhood
</th></tr>
<tr>
<td>M1A
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M2A
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3A
</td>
<td>North York
</td>
<td>Parkwoods
</td></tr>
<tr>
<td>M4A
</td>
<td>North York
</td>
<td>Victoria Village
</td></tr>
<tr>
<td>M5A
</td>
<td>Downtown Toronto
</td>
<td>Regent Park, Harbourfront
</td></tr>
<tr>
<td>M6A
</td>
<td>North York
</td>
<td>Lawrence Manor, Lawrence Heights
</td></tr>
<tr>
<td>M7A
</td>
<td>Downtown Toronto
</td>
<td>Queen's Park, Ontario Provincial Government
</td></tr>
<tr>
<td>M8A
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9A
</td>
<td>Etobicoke
</td>
<td>Islington Avenue, Humber Valley Village
</td></tr>
<tr>
<td>M1B
</td>
<td>Scarborough
</td>
<td>Malvern, Rouge
</td></tr>
<tr>
<td>M2B
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3B
</td>
<td>North York
</td>
<td>Don Mills
</td></tr>
<tr>
<td>M4B
</td>
<td>East York
</td>
<td>Parkview Hill, Woodbine Gardens
</td></tr>
<tr>
<td>M5B
</td>
<td>Downtown Toronto
</td>
<td>Garden District, Ryerson
</td></tr>
<tr>
<td>M6B
</td>
<td>North York
</td>
<td>Glencairn
</td></tr>
<tr>
<td>M7B
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8B
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9B
</td>
<td>Etobicoke
</td>
<td>West Deane Park, Princess Gardens, Martin Grove, Islington, Cloverdale
</td></tr>
<tr>
<td>M1C
</td>
<td>Scarborough
</td>
<td>Rouge Hill, Port Union, Highland Creek
</td></tr>
<tr>
<td>M2C
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3C
</td>
<td>North York
</td>
<td>Don Mills
</td></tr>
<tr>
<td>M4C
</td>
<td>East York
</td>
<td>Woodbine Heights
</td></tr>
<tr>
<td>M5C
</td>
<td>Downtown Toronto
</td>
<td>St. James Town
</td></tr>
<tr>
<td>M6C
</td>
<td>York
</td>
<td>Humewood-Cedarvale
</td></tr>
<tr>
<td>M7C
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8C
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9C
</td>
<td>Etobicoke
</td>
<td>Eringate, Bloordale Gardens, Old Burnhamthorpe, Markland Wood
</td></tr>
<tr>
<td>M1E
</td>
<td>Scarborough
</td>
<td>Guildwood, Morningside, West Hill
</td></tr>
<tr>
<td>M2E
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3E
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4E
</td>
<td>East Toronto
</td>
<td>The Beaches
</td></tr>
<tr>
<td>M5E
</td>
<td>Downtown Toronto
</td>
<td>Berczy Park
</td></tr>
<tr>
<td>M6E
</td>
<td>York
</td>
<td>Caledonia-Fairbanks
</td></tr>
<tr>
<td>M7E
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8E
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9E
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1G
</td>
<td>Scarborough
</td>
<td>Woburn
</td></tr>
<tr>
<td>M2G
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3G
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4G
</td>
<td>East York
</td>
<td>Leaside
</td></tr>
<tr>
<td>M5G
</td>
<td>Downtown Toronto
</td>
<td>Central Bay Street
</td></tr>
<tr>
<td>M6G
</td>
<td>Downtown Toronto
</td>
<td>Christie
</td></tr>
<tr>
<td>M7G
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8G
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9G
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1H
</td>
<td>Scarborough
</td>
<td>Cedarbrae
</td></tr>
<tr>
<td>M2H
</td>
<td>North York
</td>
<td>Hillcrest Village
</td></tr>
<tr>
<td>M3H
</td>
<td>North York
</td>
<td>Bathurst Manor, Wilson Heights, Downsview North
</td></tr>
<tr>
<td>M4H
</td>
<td>East York
</td>
<td>Thorncliffe Park
</td></tr>
<tr>
<td>M5H
</td>
<td>Downtown Toronto
</td>
<td>Richmond, Adelaide, King
</td></tr>
<tr>
<td>M6H
</td>
<td>West Toronto
</td>
<td>Dufferin, Dovercourt Village
</td></tr>
<tr>
<td>M7H
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8H
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9H
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1J
</td>
<td>Scarborough
</td>
<td>Scarborough Village
</td></tr>
<tr>
<td>M2J
</td>
<td>North York
</td>
<td>Fairview, Henry Farm, Oriole
</td></tr>
<tr>
<td>M3J
</td>
<td>North York
</td>
<td>Northwood Park, York University
</td></tr>
<tr>
<td>M4J
</td>
<td>East York
</td>
<td>East Toronto, Broadview North (Old East York)
</td></tr>
<tr>
<td>M5J
</td>
<td>Downtown Toronto
</td>
<td>Harbourfront East, Union Station, Toronto Islands
</td></tr>
<tr>
<td>M6J
</td>
<td>West Toronto
</td>
<td>Little Portugal, Trinity
</td></tr>
<tr>
<td>M7J
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8J
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9J
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1K
</td>
<td>Scarborough
</td>
<td>Kennedy Park, Ionview, East Birchmount Park
</td></tr>
<tr>
<td>M2K
</td>
<td>North York
</td>
<td>Bayview Village
</td></tr>
<tr>
<td>M3K
</td>
<td>North York
</td>
<td>Downsview
</td></tr>
<tr>
<td>M4K
</td>
<td>East Toronto
</td>
<td>The Danforth West, Riverdale
</td></tr>
<tr>
<td>M5K
</td>
<td>Downtown Toronto
</td>
<td>Toronto Dominion Centre, Design Exchange
</td></tr>
<tr>
<td>M6K
</td>
<td>West Toronto
</td>
<td>Brockton, Parkdale Village, Exhibition Place
</td></tr>
<tr>
<td>M7K
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8K
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9K
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1L
</td>
<td>Scarborough
</td>
<td>Golden Mile, Clairlea, Oakridge
</td></tr>
<tr>
<td>M2L
</td>
<td>North York
</td>
<td>York Mills, Silver Hills
</td></tr>
<tr>
<td>M3L
</td>
<td>North York
</td>
<td>Downsview
</td></tr>
<tr>
<td>M4L
</td>
<td>East Toronto
</td>
<td>India Bazaar, The Beaches West
</td></tr>
<tr>
<td>M5L
</td>
<td>Downtown Toronto
</td>
<td>Commerce Court, Victoria Hotel
</td></tr>
<tr>
<td>M6L
</td>
<td>North York
</td>
<td>North Park, Maple Leaf Park, Upwood Park
</td></tr>
<tr>
<td>M7L
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8L
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9L
</td>
<td>North York
</td>
<td>Humber Summit
</td></tr>
<tr>
<td>M1M
</td>
<td>Scarborough
</td>
<td>Cliffside, Cliffcrest, Scarborough Village West
</td></tr>
<tr>
<td>M2M
</td>
<td>North York
</td>
<td>Willowdale, Newtonbrook
</td></tr>
<tr>
<td>M3M
</td>
<td>North York
</td>
<td>Downsview
</td></tr>
<tr>
<td>M4M
</td>
<td>East Toronto
</td>
<td>Studio District
</td></tr>
<tr>
<td>M5M
</td>
<td>North York
</td>
<td>Bedford Park, Lawrence Manor East
</td></tr>
<tr>
<td>M6M
</td>
<td>York
</td>
<td>Del Ray, Mount Dennis, Keelsdale and Silverthorn
</td></tr>
<tr>
<td>M7M
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8M
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9M
</td>
<td>North York
</td>
<td>Humberlea, Emery
</td></tr>
<tr>
<td>M1N
</td>
<td>Scarborough
</td>
<td>Birch Cliff, Cliffside West
</td></tr>
<tr>
<td>M2N
</td>
<td>North York
</td>
<td>Willowdale, Willowdale East
</td></tr>
<tr>
<td>M3N
</td>
<td>North York
</td>
<td>Downsview
</td></tr>
<tr>
<td>M4N
</td>
<td>Central Toronto
</td>
<td>Lawrence Park
</td></tr>
<tr>
<td>M5N
</td>
<td>Central Toronto
</td>
<td>Roselawn
</td></tr>
<tr>
<td>M6N
</td>
<td>Toronto/York
</td>
<td>Runnymede, The Junction, Weston-Pellam Park, Carlton Village
</td></tr>
<tr>
<td>M7N
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8N
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9N
</td>
<td>York
</td>
<td>Weston
</td></tr>
<tr>
<td>M1P
</td>
<td>Scarborough
</td>
<td>Dorset Park, Wexford Heights, Scarborough Town Centre
</td></tr>
<tr>
<td>M2P
</td>
<td>North York
</td>
<td>York Mills West
</td></tr>
<tr>
<td>M3P
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4P
</td>
<td>Central Toronto
</td>
<td>Davisville North
</td></tr>
<tr>
<td>M5P
</td>
<td>Central Toronto
</td>
<td>Forest Hill North &amp; West, Forest Hill Road Park
</td></tr>
<tr>
<td>M6P
</td>
<td>West Toronto
</td>
<td>High Park, The Junction South
</td></tr>
<tr>
<td>M7P
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8P
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9P
</td>
<td>Etobicoke
</td>
<td>Westmount
</td></tr>
<tr>
<td>M1R
</td>
<td>Scarborough
</td>
<td>Wexford, Maryvale
</td></tr>
<tr>
<td>M2R
</td>
<td>North York
</td>
<td>Willowdale, Willowdale West
</td></tr>
<tr>
<td>M3R
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4R
</td>
<td>Central Toronto
</td>
<td>North Toronto West,  Lawrence Park
</td></tr>
<tr>
<td>M5R
</td>
<td>Central Toronto
</td>
<td>The Annex, North Midtown, Yorkville
</td></tr>
<tr>
<td>M6R
</td>
<td>West Toronto
</td>
<td>Parkdale, Roncesvalles
</td></tr>
<tr>
<td>M7R
</td>
<td>Mississauga
</td>
<td>Canada Post Gateway Processing Centre
</td></tr>
<tr>
<td>M8R
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9R
</td>
<td>Etobicoke
</td>
<td>Kingsview Village, St. Phillips, Martin Grove Gardens, Richview Gardens
</td></tr>
<tr>
<td>M1S
</td>
<td>Scarborough
</td>
<td>Agincourt
</td></tr>
<tr>
<td>M2S
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3S
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4S
</td>
<td>Central Toronto
</td>
<td>Davisville
</td></tr>
<tr>
<td>M5S
</td>
<td>Downtown Toronto
</td>
<td>University of Toronto, Harbord
</td></tr>
<tr>
<td>M6S
</td>
<td>West Toronto
</td>
<td>Runnymede, Swansea
</td></tr>
<tr>
<td>M7S
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8S
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9S
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1T
</td>
<td>Scarborough
</td>
<td>Clarks Corners, Tam O'Shanter, Sullivan
</td></tr>
<tr>
<td>M2T
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3T
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4T
</td>
<td>Central Toronto
</td>
<td>Moore Park, Summerhill East
</td></tr>
<tr>
<td>M5T
</td>
<td>Downtown Toronto
</td>
<td>Kensington Market, Chinatown, Grange Park
</td></tr>
<tr>
<td>M6T
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M7T
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8T
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M9T
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1V
</td>
<td>Scarborough
</td>
<td>Milliken, Agincourt North, Steeles East, L'Amoreaux East
</td></tr>
<tr>
<td>M2V
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3V
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4V
</td>
<td>Central Toronto
</td>
<td>Summerhill West, Rathnelly, South Hill, Forest Hill SE, Deer Park
</td></tr>
<tr>
<td>M5V
</td>
<td>Downtown Toronto
</td>
<td>CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport
</td></tr>
<tr>
<td>M6V
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M7V
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8V
</td>
<td>Etobicoke
</td>
<td>New Toronto, Mimico South, Humber Bay Shores
</td></tr>
<tr>
<td>M9V
</td>
<td>Etobicoke
</td>
<td>South Steeles, Silverstone, Humbergate, Jamestown, Mount Olive, Beaumond Heights, Thistletown, Albion Gardens
</td></tr>
<tr>
<td>M1W
</td>
<td>Scarborough
</td>
<td>Steeles West, L'Amoreaux West
</td></tr>
<tr>
<td>M2W
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3W
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4W
</td>
<td>Downtown Toronto
</td>
<td>Rosedale
</td></tr>
<tr>
<td>M5W
</td>
<td>Downtown Toronto
</td>
<td>Stn A PO Boxes
</td></tr>
<tr>
<td>M6W
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M7W
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8W
</td>
<td>Etobicoke
</td>
<td>Alderwood, Long Branch
</td></tr>
<tr>
<td>M9W
</td>
<td>Etobicoke
</td>
<td>Northwest, West Humber - Clairville
</td></tr>
<tr>
<td>M1X
</td>
<td>Scarborough
</td>
<td>Upper Rouge
</td></tr>
<tr>
<td>M2X
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3X
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4X
</td>
<td>Downtown Toronto
</td>
<td>St. James Town, Cabbagetown
</td></tr>
<tr>
<td>M5X
</td>
<td>Downtown Toronto
</td>
<td>First Canadian Place, Underground city
</td></tr>
<tr>
<td>M6X
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M7X
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8X
</td>
<td>Etobicoke
</td>
<td>The Kingsway, Montgomery Road, Old Mill North
</td></tr>
<tr>
<td>M9X
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1Y
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M2Y
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3Y
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4Y
</td>
<td>Downtown Toronto
</td>
<td>Church and Wellesley
</td></tr>
<tr>
<td>M5Y
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M6Y
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M7Y
</td>
<td>East Toronto
</td>
<td>Business reply mail Processing Centre, South Central Letter Processing Plant Toronto
</td></tr>
<tr>
<td>M8Y
</td>
<td>Etobicoke
</td>
<td>Old Mill South, King's Mill Park, Sunnylea, Humber Bay, Mimico NE, The Queensway East, Royal York South East, Kingsway Park South East
</td></tr>
<tr>
<td>M9Y
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M1Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M2Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M3Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M4Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M5Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M6Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M7Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr>
<tr>
<td>M8Z
</td>
<td>Etobicoke
</td>
<td>Mimico NW, The Queensway West, South of Bloor, Kingsway Park South West, Royal York South West
</td></tr>
<tr>
<td>M9Z
</td>
<td>Not assigned
</td>
<td>Not assigned
</td></tr></tbody></table>


**Step-3 : Converting it into Pandas DataFrame**


```python
dfs = pd.read_html(tab)
df=dfs[0]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>



**Step-4: Processing & Cleansing the Data:**


```python

df1 = df[df.Borough != 'Not assigned']    # Dropping rows where Borough is 'Not assigned'

df2 = df1.groupby(['Postal Code','Borough'], sort=False).agg(', '.join)  # Combining neighbourhoods with same Postal-code
df2.reset_index(inplace=True)

df2['Neighbourhood'] = np.where(df2['Neighbourhood'] == 'Not assigned',df2['Borough'], df2['Neighbourhood'])  # Replacing the name of the neighbourhoods which are 'Not assigned' with names of Borough

df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>M8X</td>
      <td>Etobicoke</td>
      <td>The Kingsway, Montgomery Road, Old Mill North</td>
    </tr>
    <tr>
      <th>99</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business reply mail Processing Centre, South C...</td>
    </tr>
    <tr>
      <th>101</th>
      <td>M8Y</td>
      <td>Etobicoke</td>
      <td>Old Mill South, King's Mill Park, Sunnylea, Hu...</td>
    </tr>
    <tr>
      <th>102</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Mimico NW, The Queensway West, South of Bloor,...</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 3 columns</p>
</div>



**Step-5: Shape of DataFrame:**


```python
df2.shape
```




    (103, 3)



----------------------------- **End of NoteBook (Part-1)** ---------------------------------------

# **(Part-2): Geographical coordinates of each postal code**

**Step-1: Importing the csv file conatining the latitudes and longitude**


```python
lat_longitude = pd.read_csv('https://cocl.us/Geospatial_data')
lat_longitude.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



**Step-2: Merging the 2 DataFrames**


```python
df3 = pd.merge(df2,lat_longitude,on='Postal Code')
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.753259</td>
      <td>-79.329656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td>43.725882</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
      <td>43.718518</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
  </tbody>
</table>
</div>



----------------------------- **End of NoteBook (Part-2)** ---------------------------------------

# **(Part-3): Clustering & Visualizing Neighborhood**

**Step:1 Clustering all the rows from the data frame which contains Toronto in their Borough**


```python
df_toronto = df3[df3['Borough'].str.contains('Toronto',regex=False)]
#df_toronto
df_toronto1 = df_toronto[['Borough','Neighbourhood','Latitude','Longitude']]
```

**Step-2: Visualizing all the Neighbourhoods through Folium library:**


```python
map_toronto = folium.Map(location=[43.651070,-79.347015],zoom_start=10)

for lat,lng,borough,neighbourhood in zip(df_toronto['Latitude'],df_toronto['Longitude'],df_toronto['Borough'],df_toronto['Neighbourhood']):
    label = '{}, {}'.format(neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
    [lat,lng],
    radius=5,
    popup=label,
    color='yellow',
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.8,
    parse_html=False).add_to(map_toronto)
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%3Cscript%3EL_PREFER_CANVAS%20%3D%20false%3B%20L_NO_TOUCH%20%3D%20false%3B%20L_DISABLE_3D%20%3D%20false%3B%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%20%23map_a35acece25864b8580f42108c6c6d7ec%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%20%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%20%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_a35acece25864b8580f42108c6c6d7ec%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20bounds%20%3D%20null%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_a35acece25864b8580f42108c6c6d7ec%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27map_a35acece25864b8580f42108c6c6d7ec%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7Bcenter%3A%20%5B43.65107%2C-79.347015%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2010%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxBounds%3A%20bounds%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20layers%3A%20%5B%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20worldCopyJump%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_6d4cf6b6c0ad440e85453c77bb819b8e%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22attribution%22%3A%20null%2C%0A%20%20%22detectRetina%22%3A%20false%2C%0A%20%20%22maxZoom%22%3A%2018%2C%0A%20%20%22minZoom%22%3A%201%2C%0A%20%20%22noWrap%22%3A%20false%2C%0A%20%20%22subdomains%22%3A%20%22abc%22%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ae854eee42fb4d1497b160a8685acbd8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6542599%2C-79.3606359%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3581e8bbb1b048f3ad7527a896c6fd14%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_198e9151b70449a8abd2d8c3224a86b6%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_198e9151b70449a8abd2d8c3224a86b6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERegent%20Park%2C%20Harbourfront%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3581e8bbb1b048f3ad7527a896c6fd14.setContent%28html_198e9151b70449a8abd2d8c3224a86b6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ae854eee42fb4d1497b160a8685acbd8.bindPopup%28popup_3581e8bbb1b048f3ad7527a896c6fd14%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ba253f95074941738e9705230b384e28%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6623015%2C-79.3894938%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3149f1ed123e43a881476a523ce42eff%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6abb1081c33c4f41bc93bffb61c40d43%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6abb1081c33c4f41bc93bffb61c40d43%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueen%26%2339%3Bs%20Park%2C%20Ontario%20Provincial%20Government%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3149f1ed123e43a881476a523ce42eff.setContent%28html_6abb1081c33c4f41bc93bffb61c40d43%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ba253f95074941738e9705230b384e28.bindPopup%28popup_3149f1ed123e43a881476a523ce42eff%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1d4227d510bc49398cf5d826a9a56e4a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6571618%2C-79.3789371%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_dfd124d1b9c54b2cb106f01f0c8b2230%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0918339650754c709098a868d2a35d3b%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_0918339650754c709098a868d2a35d3b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGarden%20District%2C%20Ryerson%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_dfd124d1b9c54b2cb106f01f0c8b2230.setContent%28html_0918339650754c709098a868d2a35d3b%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_1d4227d510bc49398cf5d826a9a56e4a.bindPopup%28popup_dfd124d1b9c54b2cb106f01f0c8b2230%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b1a96e405000461d9ec160f76f165c73%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6514939%2C-79.3754179%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_371d3378287c482b8e181ae3f7e22881%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2a43a823b0e84a49a175f6891af770ce%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2a43a823b0e84a49a175f6891af770ce%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_371d3378287c482b8e181ae3f7e22881.setContent%28html_2a43a823b0e84a49a175f6891af770ce%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b1a96e405000461d9ec160f76f165c73.bindPopup%28popup_371d3378287c482b8e181ae3f7e22881%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9554ba090f2244d5bf10908f7467539a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6763574%2C-79.2930312%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_15b8890287cf4960abe9bb0e8bd859ff%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3f383ee9003a477e94318cabb6849664%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3f383ee9003a477e94318cabb6849664%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Beaches%2C%20East%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_15b8890287cf4960abe9bb0e8bd859ff.setContent%28html_3f383ee9003a477e94318cabb6849664%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9554ba090f2244d5bf10908f7467539a.bindPopup%28popup_15b8890287cf4960abe9bb0e8bd859ff%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_eda193c3215d4b2283b2d4fd25861e6e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6447708%2C-79.3733064%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_cbc2c379bad840668fea9bd3bf52a525%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d68e1dc99f604c7694354d550ef1b7b1%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d68e1dc99f604c7694354d550ef1b7b1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBerczy%20Park%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_cbc2c379bad840668fea9bd3bf52a525.setContent%28html_d68e1dc99f604c7694354d550ef1b7b1%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_eda193c3215d4b2283b2d4fd25861e6e.bindPopup%28popup_cbc2c379bad840668fea9bd3bf52a525%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_621ad7796b2e41b9959c5a270092c16a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6579524%2C-79.3873826%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_19722b157bfd4dec80592b38a0fd7288%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_83b6e92848aa4dbcb7f6c661cd568c84%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_83b6e92848aa4dbcb7f6c661cd568c84%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Bay%20Street%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_19722b157bfd4dec80592b38a0fd7288.setContent%28html_83b6e92848aa4dbcb7f6c661cd568c84%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_621ad7796b2e41b9959c5a270092c16a.bindPopup%28popup_19722b157bfd4dec80592b38a0fd7288%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fdbcd3fc4dc14dc79649a83f7e9c4e55%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.669542%2C-79.4225637%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_62b387ad88cf4f27854c19d2566a642f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_27ead7f35d984157b673643fbba3a1c7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_27ead7f35d984157b673643fbba3a1c7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChristie%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_62b387ad88cf4f27854c19d2566a642f.setContent%28html_27ead7f35d984157b673643fbba3a1c7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fdbcd3fc4dc14dc79649a83f7e9c4e55.bindPopup%28popup_62b387ad88cf4f27854c19d2566a642f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7d79cd28cf06431389c463477549bcc1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6505712%2C-79.3845675%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a32f1f20458b4a23840cae9359188069%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9c1b28b2465a477396729a317258da9a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9c1b28b2465a477396729a317258da9a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%2C%20Adelaide%2C%20King%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a32f1f20458b4a23840cae9359188069.setContent%28html_9c1b28b2465a477396729a317258da9a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_7d79cd28cf06431389c463477549bcc1.bindPopup%28popup_a32f1f20458b4a23840cae9359188069%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_643c2238ec104be7aeee261e693bbdbb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6690051%2C-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0b42edd8e709409899bf8494df93b999%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3e409a09ad844511ac47146bcb4a2dbd%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3e409a09ad844511ac47146bcb4a2dbd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDufferin%2C%20Dovercourt%20Village%2C%20West%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0b42edd8e709409899bf8494df93b999.setContent%28html_3e409a09ad844511ac47146bcb4a2dbd%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_643c2238ec104be7aeee261e693bbdbb.bindPopup%28popup_0b42edd8e709409899bf8494df93b999%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4fe45ba5b2b54c148d006d9fa84a0b96%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6408157%2C-79.3817523%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_11f16c35b09d49c09b9b8604cd39f681%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_549721b1652547abbdc5c95194d1c8e4%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_549721b1652547abbdc5c95194d1c8e4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHarbourfront%20East%2C%20Union%20Station%2C%20Toronto%20Islands%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_11f16c35b09d49c09b9b8604cd39f681.setContent%28html_549721b1652547abbdc5c95194d1c8e4%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_4fe45ba5b2b54c148d006d9fa84a0b96.bindPopup%28popup_11f16c35b09d49c09b9b8604cd39f681%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_585b311e40984c0b85339d941681dd29%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6479267%2C-79.4197497%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9867ccdd3fb848aa86be2e86e4c031bb%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_343a55f94b324a7dac5e2fcbe2c2858e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_343a55f94b324a7dac5e2fcbe2c2858e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELittle%20Portugal%2C%20Trinity%2C%20West%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9867ccdd3fb848aa86be2e86e4c031bb.setContent%28html_343a55f94b324a7dac5e2fcbe2c2858e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_585b311e40984c0b85339d941681dd29.bindPopup%28popup_9867ccdd3fb848aa86be2e86e4c031bb%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_73a9ec1bb4f74072be6c91c4d3569219%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6795571%2C-79.352188%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a01d98d1f19448fea8556d8519cd4267%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_686f5fc35ace4ee6a9ec33a7a8a813ff%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_686f5fc35ace4ee6a9ec33a7a8a813ff%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Danforth%20West%2C%20Riverdale%2C%20East%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a01d98d1f19448fea8556d8519cd4267.setContent%28html_686f5fc35ace4ee6a9ec33a7a8a813ff%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_73a9ec1bb4f74072be6c91c4d3569219.bindPopup%28popup_a01d98d1f19448fea8556d8519cd4267%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_34b9570cbcf0414988476a05e50cc5e9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6471768%2C-79.3815764%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c0102dcefcdf47a387bbb775aa435373%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_559fe98fd367478d9af5d2e952991920%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_559fe98fd367478d9af5d2e952991920%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EToronto%20Dominion%20Centre%2C%20Design%20Exchange%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c0102dcefcdf47a387bbb775aa435373.setContent%28html_559fe98fd367478d9af5d2e952991920%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_34b9570cbcf0414988476a05e50cc5e9.bindPopup%28popup_c0102dcefcdf47a387bbb775aa435373%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ed10feb5ece5469aabc7e1eacfd14716%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6368472%2C-79.4281914%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_6da43f0300bc42869777da81be08f91c%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_28000f933f6a4cf49e2c8a777898ae60%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_28000f933f6a4cf49e2c8a777898ae60%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrockton%2C%20Parkdale%20Village%2C%20Exhibition%20Place%2C%20West%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_6da43f0300bc42869777da81be08f91c.setContent%28html_28000f933f6a4cf49e2c8a777898ae60%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ed10feb5ece5469aabc7e1eacfd14716.bindPopup%28popup_6da43f0300bc42869777da81be08f91c%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_81695d47453246f780b09e01d0b09607%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6689985%2C-79.3155716%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_63cf1d5fff30462fbf9a00f7e33e9ba2%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5668028594634e51a424eb25c2b59326%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5668028594634e51a424eb25c2b59326%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIndia%20Bazaar%2C%20The%20Beaches%20West%2C%20East%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_63cf1d5fff30462fbf9a00f7e33e9ba2.setContent%28html_5668028594634e51a424eb25c2b59326%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_81695d47453246f780b09e01d0b09607.bindPopup%28popup_63cf1d5fff30462fbf9a00f7e33e9ba2%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fc04c498e36d45b5ad867b3c4969c9e0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6481985%2C-79.3798169%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e100e0784e9e408d9bf1e24c64ffe31f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ca7fc61373f048af9e3e5f3c57de96dd%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ca7fc61373f048af9e3e5f3c57de96dd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECommerce%20Court%2C%20Victoria%20Hotel%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e100e0784e9e408d9bf1e24c64ffe31f.setContent%28html_ca7fc61373f048af9e3e5f3c57de96dd%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fc04c498e36d45b5ad867b3c4969c9e0.bindPopup%28popup_e100e0784e9e408d9bf1e24c64ffe31f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0790c524ca7e4d41991bc74967a58b01%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6595255%2C-79.340923%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_61d6cde050554f11866c54e7a994d6ec%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4a1f0c5069f14393824189109423162a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_4a1f0c5069f14393824189109423162a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStudio%20District%2C%20East%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_61d6cde050554f11866c54e7a994d6ec.setContent%28html_4a1f0c5069f14393824189109423162a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0790c524ca7e4d41991bc74967a58b01.bindPopup%28popup_61d6cde050554f11866c54e7a994d6ec%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_12c87e2906eb423cad652a70611a2985%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7280205%2C-79.3887901%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_54710769f2974b4381e446a2b74d0879%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_517f634fbabb4fe9a3987d0fa74f2693%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_517f634fbabb4fe9a3987d0fa74f2693%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Park%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_54710769f2974b4381e446a2b74d0879.setContent%28html_517f634fbabb4fe9a3987d0fa74f2693%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_12c87e2906eb423cad652a70611a2985.bindPopup%28popup_54710769f2974b4381e446a2b74d0879%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b53cc7038cfb4022ab4697746e86a1a9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7116948%2C-79.4169356%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d6332df51cf6471da9f4cb8613722ab5%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1d66eb2f5db14e009a4cadae1571a134%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1d66eb2f5db14e009a4cadae1571a134%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERoselawn%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d6332df51cf6471da9f4cb8613722ab5.setContent%28html_1d66eb2f5db14e009a4cadae1571a134%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b53cc7038cfb4022ab4697746e86a1a9.bindPopup%28popup_d6332df51cf6471da9f4cb8613722ab5%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ce8f56ccc20b48cc8578c2137bd89906%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6731853%2C-79.4872619%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e350d0e8d0ff439dae18a7511c2610aa%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_27ac6f1b0f41485d8122d6b4cc43bdb6%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_27ac6f1b0f41485d8122d6b4cc43bdb6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20The%20Junction%2C%20Weston-Pellam%20Park%2C%20Carlton%20Village%2C%20Toronto/York%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e350d0e8d0ff439dae18a7511c2610aa.setContent%28html_27ac6f1b0f41485d8122d6b4cc43bdb6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ce8f56ccc20b48cc8578c2137bd89906.bindPopup%28popup_e350d0e8d0ff439dae18a7511c2610aa%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_269c68f35c4c475cbb745128093c7388%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7127511%2C-79.3901975%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0f9b9d07698641b1ab7e725e9adafc0f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_686427c9f14948a493f611c5ed28ea12%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_686427c9f14948a493f611c5ed28ea12%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20North%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0f9b9d07698641b1ab7e725e9adafc0f.setContent%28html_686427c9f14948a493f611c5ed28ea12%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_269c68f35c4c475cbb745128093c7388.bindPopup%28popup_0f9b9d07698641b1ab7e725e9adafc0f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_11936e7747e449c0b15397170c5537eb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6969476%2C-79.4113072%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_5ba3373150e04673beeee614508400b3%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3fe43971798b46fc9a03955e54b014e3%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3fe43971798b46fc9a03955e54b014e3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EForest%20Hill%20North%20%26amp%3B%20West%2C%20Forest%20Hill%20Road%20Park%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_5ba3373150e04673beeee614508400b3.setContent%28html_3fe43971798b46fc9a03955e54b014e3%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_11936e7747e449c0b15397170c5537eb.bindPopup%28popup_5ba3373150e04673beeee614508400b3%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fef9831947be4d3a868905d2222c24df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6616083%2C-79.4647633%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_51cbb845b952463689b43895df74f265%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cb6c72eaa40a4cc3ba0b28ee2d154571%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_cb6c72eaa40a4cc3ba0b28ee2d154571%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHigh%20Park%2C%20The%20Junction%20South%2C%20West%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_51cbb845b952463689b43895df74f265.setContent%28html_cb6c72eaa40a4cc3ba0b28ee2d154571%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fef9831947be4d3a868905d2222c24df.bindPopup%28popup_51cbb845b952463689b43895df74f265%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_87b7c4a9cb82492b8f5e9573d3378480%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7153834%2C-79.4056784%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_daa667ead41345a389ee1a1b53bf0213%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9e7380172f9a4428ba1df90cc1a39b78%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9e7380172f9a4428ba1df90cc1a39b78%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Toronto%20West%2C%20Lawrence%20Park%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_daa667ead41345a389ee1a1b53bf0213.setContent%28html_9e7380172f9a4428ba1df90cc1a39b78%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_87b7c4a9cb82492b8f5e9573d3378480.bindPopup%28popup_daa667ead41345a389ee1a1b53bf0213%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_609b10fb0b7742818d7733f83d3e90ec%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6727097%2C-79.4056784%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_0bc421a647f84e559344458ee178a061%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6d37781c61d24b5a93219e110aea99be%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6d37781c61d24b5a93219e110aea99be%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Annex%2C%20North%20Midtown%2C%20Yorkville%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_0bc421a647f84e559344458ee178a061.setContent%28html_6d37781c61d24b5a93219e110aea99be%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_609b10fb0b7742818d7733f83d3e90ec.bindPopup%28popup_0bc421a647f84e559344458ee178a061%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ebf6c8df06c14e71b3d3601dcbe3c8f6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6489597%2C-79.456325%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ec176fda619b48f5ad4561b94becf75a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a192a7efa5484468bde166ceb6cb8cf2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a192a7efa5484468bde166ceb6cb8cf2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkdale%2C%20Roncesvalles%2C%20West%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ec176fda619b48f5ad4561b94becf75a.setContent%28html_a192a7efa5484468bde166ceb6cb8cf2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ebf6c8df06c14e71b3d3601dcbe3c8f6.bindPopup%28popup_ec176fda619b48f5ad4561b94becf75a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_291d857221cf4c4dad76aa353fd65e2f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7043244%2C-79.3887901%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_827b1c9283424fa682ee72f341949051%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e14fa48dd1334a9ab7164686457ff973%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_e14fa48dd1334a9ab7164686457ff973%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_827b1c9283424fa682ee72f341949051.setContent%28html_e14fa48dd1334a9ab7164686457ff973%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_291d857221cf4c4dad76aa353fd65e2f.bindPopup%28popup_827b1c9283424fa682ee72f341949051%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_54c4b2036882412a9487981f4431cd9d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6626956%2C-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c69918a6707748abbc8783efecbceda6%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_10cf75fedbbc4b5ebe02aa3ed543e69c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_10cf75fedbbc4b5ebe02aa3ed543e69c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUniversity%20of%20Toronto%2C%20Harbord%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c69918a6707748abbc8783efecbceda6.setContent%28html_10cf75fedbbc4b5ebe02aa3ed543e69c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_54c4b2036882412a9487981f4431cd9d.bindPopup%28popup_c69918a6707748abbc8783efecbceda6%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5414f621ce28421b8612c5f9b80ccf85%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6515706%2C-79.4844499%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_315ff9f9448444da8b89b2d3970498a7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c8dbd01da0974bf7a4ec9ba228a5da5f%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c8dbd01da0974bf7a4ec9ba228a5da5f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20Swansea%2C%20West%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_315ff9f9448444da8b89b2d3970498a7.setContent%28html_c8dbd01da0974bf7a4ec9ba228a5da5f%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5414f621ce28421b8612c5f9b80ccf85.bindPopup%28popup_315ff9f9448444da8b89b2d3970498a7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d1369c3bc2f74ceea5dadcdda3a05eea%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6895743%2C-79.3831599%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_286949256e3640d9a2be634d1e630ece%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9a7f3e4ffd2742d2a56117b2f441d045%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9a7f3e4ffd2742d2a56117b2f441d045%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMoore%20Park%2C%20Summerhill%20East%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_286949256e3640d9a2be634d1e630ece.setContent%28html_9a7f3e4ffd2742d2a56117b2f441d045%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d1369c3bc2f74ceea5dadcdda3a05eea.bindPopup%28popup_286949256e3640d9a2be634d1e630ece%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b57f30516b4f48939b84d1553593eff9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6532057%2C-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e5c6a6958c33490dae66c4fe831fb3df%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6a69a08857fe4c67a038cc77f22a02f7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6a69a08857fe4c67a038cc77f22a02f7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKensington%20Market%2C%20Chinatown%2C%20Grange%20Park%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e5c6a6958c33490dae66c4fe831fb3df.setContent%28html_6a69a08857fe4c67a038cc77f22a02f7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b57f30516b4f48939b84d1553593eff9.bindPopup%28popup_e5c6a6958c33490dae66c4fe831fb3df%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a7c96724e7ab4219823a66bf1db471d5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6864123%2C-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_033dc47e79194958b76762022fd93f27%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1dc4f6b6cbe743ccaaae2b4d83415120%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1dc4f6b6cbe743ccaaae2b4d83415120%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESummerhill%20West%2C%20Rathnelly%2C%20South%20Hill%2C%20Forest%20Hill%20SE%2C%20Deer%20Park%2C%20Central%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_033dc47e79194958b76762022fd93f27.setContent%28html_1dc4f6b6cbe743ccaaae2b4d83415120%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a7c96724e7ab4219823a66bf1db471d5.bindPopup%28popup_033dc47e79194958b76762022fd93f27%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f237ba8ba81647d5873a956d9e2f36f1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6289467%2C-79.3944199%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_b9d4bc2d8f2a44ee95c9b8bb99aa2c7c%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c4c923badd544277a36dad48678c75cd%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_c4c923badd544277a36dad48678c75cd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECN%20Tower%2C%20King%20and%20Spadina%2C%20Railway%20Lands%2C%20Harbourfront%20West%2C%20Bathurst%20Quay%2C%20South%20Niagara%2C%20Island%20airport%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_b9d4bc2d8f2a44ee95c9b8bb99aa2c7c.setContent%28html_c4c923badd544277a36dad48678c75cd%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_f237ba8ba81647d5873a956d9e2f36f1.bindPopup%28popup_b9d4bc2d8f2a44ee95c9b8bb99aa2c7c%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f4effbcb9b204c85b66573edc90455fb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6795626%2C-79.3775294%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2dfb4e1531cd4c5f9beac1eb3b365339%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_39e55cf51e40478190af9d113c836374%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_39e55cf51e40478190af9d113c836374%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERosedale%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2dfb4e1531cd4c5f9beac1eb3b365339.setContent%28html_39e55cf51e40478190af9d113c836374%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_f4effbcb9b204c85b66573edc90455fb.bindPopup%28popup_2dfb4e1531cd4c5f9beac1eb3b365339%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7c5f1d41292049228ad317496eca3a12%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6464352%2C-79.374846%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f9153f830bd24cdd887c31dcffe8e83f%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8b277296fa624324a9da1cc420501da2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_8b277296fa624324a9da1cc420501da2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStn%20A%20PO%20Boxes%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f9153f830bd24cdd887c31dcffe8e83f.setContent%28html_8b277296fa624324a9da1cc420501da2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_7c5f1d41292049228ad317496eca3a12.bindPopup%28popup_f9153f830bd24cdd887c31dcffe8e83f%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ac2be871be594b4eb2173e925a8d587f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.667967%2C-79.3676753%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3a1cd05f00744929a813b01c73d09200%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3fa0c0c2153b445298cff1bb800c8057%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3fa0c0c2153b445298cff1bb800c8057%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Cabbagetown%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3a1cd05f00744929a813b01c73d09200.setContent%28html_3fa0c0c2153b445298cff1bb800c8057%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_ac2be871be594b4eb2173e925a8d587f.bindPopup%28popup_3a1cd05f00744929a813b01c73d09200%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6e4b5b27c3714105a31fe68572ead5f1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6484292%2C-79.3822802%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2db25d492b4140e695b14e9665a3ab40%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b9c36865a73443418318038c0e4be16e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b9c36865a73443418318038c0e4be16e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFirst%20Canadian%20Place%2C%20Underground%20city%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2db25d492b4140e695b14e9665a3ab40.setContent%28html_b9c36865a73443418318038c0e4be16e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_6e4b5b27c3714105a31fe68572ead5f1.bindPopup%28popup_2db25d492b4140e695b14e9665a3ab40%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0021c724c3b2497cb163b22fc7f6fcbc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6658599%2C-79.3831599%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_fe19ee6124d7420aaad0a630bc3dab33%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_129e49f8a2ad45dd8c32b1f29cdf3c17%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_129e49f8a2ad45dd8c32b1f29cdf3c17%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChurch%20and%20Wellesley%2C%20Downtown%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_fe19ee6124d7420aaad0a630bc3dab33.setContent%28html_129e49f8a2ad45dd8c32b1f29cdf3c17%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0021c724c3b2497cb163b22fc7f6fcbc.bindPopup%28popup_fe19ee6124d7420aaad0a630bc3dab33%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b2d533732b49448b8d2a70ba15b50b6c%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6627439%2C-79.321558%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22yellow%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22%233186cc%22%2C%0A%20%20%22fillOpacity%22%3A%200.8%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_a35acece25864b8580f42108c6c6d7ec%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_c0f30b5b205c4c73971cfe3869d057ae%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7b9924b5d1a244a2a8b3a2f5150aee8a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_7b9924b5d1a244a2a8b3a2f5150aee8a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBusiness%20reply%20mail%20Processing%20Centre%2C%20South%20Central%20Letter%20Processing%20Plant%20Toronto%2C%20East%20Toronto%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_c0f30b5b205c4c73971cfe3869d057ae.setContent%28html_7b9924b5d1a244a2a8b3a2f5150aee8a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b2d533732b49448b8d2a70ba15b50b6c.bindPopup%28popup_c0f30b5b205c4c73971cfe3869d057ae%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python

```

----------------------------- **End of NoteBook (Part-3)** ---------------------------------------


```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('Number of Neighborhood for each Borough in Toronto')
#On x-axis
plt.xlabel('Borough', fontsize = 15)
#On y-axis
plt.ylabel('No.of Neighborhood', fontsize=15)
#giving a bar plot
df_toronto.groupby('Borough')['Neighbourhood'].count().plot(kind='bar')
#legend
plt.legend()
#displays the plot
plt.show()
```


![png](output_41_0.png)


# **Define Foursquare Credentials and Version**

**Step-1 : Defining credentials:**


```python
CLIENT_ID = 'FI0TP43OOSYSUVVFRFHQAIF5YIOJLE0TBY4044WOTZRZVBLI' # your Foursquare ID
CLIENT_SECRET = 'FDEKE3FMRHJU1PWN3O00GIVZ2CGRQPK52KDYRTJEFXUYNUFR' # your Foursquare Secret
ACCESS_TOKEN = 'I2AMKTPDM0CTIGRK4DXN232OU1W4JT5HSCCAUFEQAJMRYV2R' # your FourSquare Access Token
VERSION = '20210321'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: FI0TP43OOSYSUVVFRFHQAIF5YIOJLE0TBY4044WOTZRZVBLI
    CLIENT_SECRET:FDEKE3FMRHJU1PWN3O00GIVZ2CGRQPK52KDYRTJEFXUYNUFR



```python
address = 'Toronto,Canada'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)

search_query = 'Coffee'
radius = 10000
print(search_query + ' .... OK!')

```

    43.6534817 -79.3839347
    Coffee .... OK!


**We will define url for getting the coffee shop venues**


```python
url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&oauth_token={}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude,ACCESS_TOKEN, VERSION, search_query, radius, LIMIT)
url
results = requests.get(url).json()
results
```




    {'meta': {'code': 200, 'requestId': '605786dc8a4a017d77997be5'},
     'notifications': [{'type': 'notificationTray', 'item': {'unreadCount': 0}}],
     'response': {'venues': [{'id': '59f784dd28122f14f9d5d63d',
        'name': 'HotBlack Coffee',
        'location': {'address': '245 Queen Street West',
         'crossStreet': 'at St Patrick St',
         'lat': 43.65036434800487,
         'lng': -79.38866907575726,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65036434800487,
           'lng': -79.38866907575726}],
         'distance': 515,
         'postalCode': 'M5V 1Z4',
         'cc': 'CA',
         'neighborhood': 'Entertainment District',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['245 Queen Street West (at St Patrick St)',
          'Toronto ON M5V 1Z4',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'venuePage': {'id': '463001529'},
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4b44fc77f964a520cc0026e3',
        'name': "Timothy's World Coffee",
        'location': {'address': '427 University Avenue',
         'lat': 43.65405317976302,
         'lng': -79.38808999785911,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65405317976302,
           'lng': -79.38808999785911}],
         'distance': 340,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['427 University Avenue', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4b0aaa8ef964a520272623e3',
        'name': "Timothy's World Coffee",
        'location': {'address': '483 Bay St, Bell Trinity Square',
         'crossStreet': 'Bell Trinity Square',
         'lat': 43.653436,
         'lng': -79.382314,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.653436,
           'lng': -79.382314}],
         'distance': 130,
         'postalCode': 'M5G 2C9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['483 Bay St, Bell Trinity Square (Bell Trinity Square)',
          'Toronto ON M5G 2C9',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4fb13c20e4b011e6f93513c0',
        'name': "Balzac's Coffee",
        'location': {'address': '122 Bond Street',
         'crossStreet': 'at Gould St.',
         'lat': 43.65785440672277,
         'lng': -79.37919981155157,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65785440672277,
           'lng': -79.37919981155157}],
         'distance': 618,
         'postalCode': 'M5B 1X8',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['122 Bond Street (at Gould St.)',
          'Toronto ON M5B 1X8',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4baa9f6cf964a520817a3ae3',
        'name': "Timothy's World Coffee",
        'location': {'address': '401 Bay St.',
         'crossStreet': 'at Richmond St. W',
         'lat': 43.65213455850074,
         'lng': -79.38117224696582,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65213455850074,
           'lng': -79.38117224696582}],
         'distance': 268,
         'postalCode': 'M5H 2Y4',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['401 Bay St. (at Richmond St. W)',
          'Toronto ON M5H 2Y4',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '53e8acc4498ee294fb100183',
        'name': "Timothy's World Coffee",
        'location': {'address': '425 University Ave',
         'crossStreet': 'Dundas',
         'lat': 43.65427,
         'lng': -79.387448,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65427,
           'lng': -79.387448}],
         'distance': 296,
         'postalCode': 'M5G 1T6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['425 University Ave (Dundas)',
          'Toronto ON M5G 1T6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4baa31def964a52037523ae3',
        'name': 'Coffee office',
        'location': {'address': '350 Bay St - 7th Floor',
         'lat': 43.649498,
         'lng': -79.386479,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.649498,
           'lng': -79.386479}],
         'distance': 488,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['350 Bay St - 7th Floor', 'Toronto ON', 'Canada']},
        'categories': [],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4fff1f96e4b042ae8acddca5',
        'name': 'Fahrenheit Coffee',
        'location': {'address': '120 Lombard St',
         'crossStreet': 'at Jarvis St',
         'lat': 43.65238358726612,
         'lng': -79.37271903848271,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65238358726612,
           'lng': -79.37271903848271}],
         'distance': 911,
         'postalCode': 'M5C 3H5',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['120 Lombard St (at Jarvis St)',
          'Toronto ON M5C 3H5',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4ec514ec9911232436e364af',
        'name': "Timothy's World Coffee",
        'location': {'address': 'Yonge',
         'crossStreet': 'Dundas',
         'lat': 43.65669995833159,
         'lng': -79.37994058195848,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65669995833159,
           'lng': -79.37994058195848}],
         'distance': 481,
         'postalCode': 'M5B 2G9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['Yonge (Dundas)', 'Toronto ON M5B 2G9', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4fccaa8fe4b05a98df3d9417',
        'name': 'Sam James Coffee Bar (SJCB)',
        'location': {'address': '150 King St. W',
         'crossStreet': 'in the PATH',
         'lat': 43.64788137014028,
         'lng': -79.38433152836829,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64788137014028,
           'lng': -79.38433152836829}],
         'distance': 624,
         'postalCode': 'M5H 4B6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['150 King St. W (in the PATH)',
          'Toronto ON M5H 4B6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16d941735',
          'name': 'Café',
          'pluralName': 'Cafés',
          'shortName': 'Café',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '536fd522498e09c6690800e2',
        'name': "Balzac's Coffee",
        'location': {'address': '10 Market Street',
         'crossStreet': 'btwn The Esplanade & Front St. E.',
         'lat': 43.64845650131932,
         'lng': -79.37178993724407,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64845650131932,
           'lng': -79.37178993724407}],
         'distance': 1126,
         'postalCode': 'M5E 1M6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['10 Market Street (btwn The Esplanade & Front St. E.)',
          'Toronto ON M5E 1M6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16d941735',
          'name': 'Café',
          'pluralName': 'Cafés',
          'shortName': 'Café',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4ad79243f964a5204c0c21e3',
        'name': 'Jetfuel Coffee',
        'location': {'address': '519 Parliament St.',
         'crossStreet': 'btwn Carlton & Winchester',
         'lat': 43.66529519392083,
         'lng': -79.3683345416816,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.66529519392083,
           'lng': -79.3683345416816}],
         'distance': 1818,
         'postalCode': 'M4X 1P3',
         'cc': 'CA',
         'neighborhood': 'Cabbagetown',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['519 Parliament St. (btwn Carlton & Winchester)',
          'Toronto ON M4X 1P3',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '563d2f2dcd10bcf27ae37c3b',
        'name': 'Pilot Coffee Roasters',
        'location': {'address': '65 Front St W',
         'crossStreet': 'btwn Bay St & York St',
         'lat': 43.64501814464698,
         'lng': -79.3804150931199,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64501814464698,
           'lng': -79.3804150931199}],
         'distance': 983,
         'postalCode': 'M5J 1E6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['65 Front St W (btwn Bay St & York St)',
          'Toronto ON M5J 1E6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '5c86b682da2e00002cf95781',
        'name': 'Second Cup Coffee Co. featuring Pinkberry Frozen Yogurt',
        'location': {'address': '600 University Avenue, Room #202',
         'lat': 43.657473,
         'lng': -79.390637,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.657473,
           'lng': -79.390637}],
         'distance': 699,
         'postalCode': 'M5G 1X5',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['600 University Avenue, Room #202',
          'Toronto ON M5G 1X5',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16d941735',
          'name': 'Café',
          'pluralName': 'Cafés',
          'shortName': 'Café',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4d261e1e3c84b1f78bf70847',
        'name': "Timothy's World Coffee",
        'location': {'address': '30 Adelaide St E',
         'lat': 43.650948,
         'lng': -79.376825,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.650948,
           'lng': -79.376825}],
         'distance': 638,
         'postalCode': 'M5C 3G8',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['30 Adelaide St E', 'Toronto ON M5C 3G8', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '569e7814498e1a7f3e01bfe4',
        'name': 'Rooster Coffee House',
        'location': {'address': '568 Jarvis St',
         'crossStreet': 'At Charles St E',
         'lat': 43.66965378571954,
         'lng': -79.379870566686,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.66965378571954,
           'lng': -79.379870566686}],
         'distance': 1829,
         'postalCode': 'M4Y 1N6',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['568 Jarvis St (At Charles St E)',
          'Toronto ON M4Y 1N6',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4bce5e21cc8cd13a7359c4cf',
        'name': "Timothy's World Coffee",
        'location': {'address': '444 Yonge St',
         'crossStreet': 'in College Park',
         'lat': 43.66046739684086,
         'lng': -79.38465356826782,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.66046739684086,
           'lng': -79.38465356826782}],
         'distance': 779,
         'postalCode': 'M5B 2H4',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['444 Yonge St (in College Park)',
          'Toronto ON M5B 2H4',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4d8a7096d85f3704d05afedb',
        'name': 'T.A.N. Coffee',
        'location': {'address': '37 Baldwin St',
         'crossStreet': 'Henry St',
         'lat': 43.65602860741956,
         'lng': -79.39353447011945,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65602860741956,
           'lng': -79.39353447011945}],
         'distance': 823,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['37 Baldwin St (Henry St)', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4ba37627f964a520263f38e3',
        'name': "Timothy's World Coffee",
        'location': {'address': '66 Wellington Street West',
         'crossStreet': 'TD Center Concourse',
         'lat': 43.64713049355658,
         'lng': -79.38077635642868,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64713049355658,
           'lng': -79.38077635642868}],
         'distance': 751,
         'postalCode': 'M5K 1A1',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['66 Wellington Street West (TD Center Concourse)',
          'Toronto ON M5K 1A1',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '51438b33e4b0a40e33fe5e77',
        'name': "Jimmy's Coffee",
        'location': {'address': '191 Baldwin St',
         'crossStreet': 'Kensington Market',
         'lat': 43.65449315540114,
         'lng': -79.40131090393002,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65449315540114,
           'lng': -79.40131090393002}],
         'distance': 1404,
         'cc': 'CA',
         'neighborhood': 'Kensington Market',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['191 Baldwin St (Kensington Market)',
          'Toronto ON',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16d941735',
          'name': 'Café',
          'pluralName': 'Cafés',
          'shortName': 'Café',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '5553954a498e8e11bc49ecf2',
        'name': 'Sam James Coffee Bar (SJCB)',
        'location': {'address': '15 Toronto Street',
         'lat': 43.65031871629752,
         'lng': -79.37621650642859,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65031871629752,
           'lng': -79.37621650642859}],
         'distance': 714,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['15 Toronto Street', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4b156e98f964a520cbac23e3',
        'name': "Timothy's World Coffee",
        'location': {'address': '801 Bay St',
         'crossStreet': 'at College St',
         'lat': 43.66071353922905,
         'lng': -79.38549125817697,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.66071353922905,
           'lng': -79.38549125817697}],
         'distance': 814,
         'postalCode': 'M5S 1Y9',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['801 Bay St (at College St)',
          'Toronto ON M5S 1Y9',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '557f05ca498ec78ac7b29315',
        'name': "Balzac's Coffee",
        'location': {'address': '7 Station St',
         'crossStreet': 'at SkyWalk',
         'lat': 43.644372584148364,
         'lng': -79.38306470027423,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.644372584148364,
           'lng': -79.38306470027423}],
         'distance': 1016,
         'postalCode': 'M5J 1C3',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['7 Station St (at SkyWalk)',
          'Toronto ON M5J 1C3',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '5c86b6b9cb3fd2002cd9c9a2',
        'name': 'Second Cup Coffee Co. featuring Pinkberry Frozen Yogurt',
        'location': {'address': '179 College Street',
         'lat': 43.65887244042741,
         'lng': -79.39415829049284,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65887244042741,
           'lng': -79.39415829049284}],
         'distance': 1018,
         'postalCode': 'M5T 1P7',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['179 College Street',
          'Toronto ON M5T 1P7',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d16d941735',
          'name': 'Café',
          'pluralName': 'Cafés',
          'shortName': 'Café',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '584593c2ebf0284fe7b103cb',
        'name': 'Fahrenheit Coffee',
        'location': {'address': '529 Richmond St W',
         'lat': 43.64703669923361,
         'lng': -79.40087579760974,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.64703669923361,
           'lng': -79.40087579760974}],
         'distance': 1541,
         'cc': 'CA',
         'neighborhood': 'Fashion District',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['529 Richmond St W', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4b9e7808f964a52091e636e3',
        'name': 'Second Cup Coffee Co.',
        'location': {'address': '200 Front St W',
         'crossStreet': 'in Simcoe Place',
         'lat': 43.645009412720164,
         'lng': -79.38581150464259,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.645009412720164,
           'lng': -79.38581150464259}],
         'distance': 955,
         'postalCode': 'M5V 3K2',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['200 Front St W (in Simcoe Place)',
          'Toronto ON M5V 3K2',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4adb5a00f964a5204c2621e3',
        'name': 'I Deal Coffee',
        'location': {'address': '84 Nassau Street',
         'lat': 43.65505778213131,
         'lng': -79.40325401691655,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65505778213131,
           'lng': -79.40325401691655}],
         'distance': 1565,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['84 Nassau Street', 'Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4b758da6f964a520cb132ee3',
        'name': 'Luba’s Coffee & Tea Boutique',
        'location': {'lat': 43.6490525948081,
         'lng': -79.37198136360134,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.6490525948081,
           'lng': -79.37198136360134}],
         'distance': 1081,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '4c3de34c7d002d7fe460b018',
        'name': "Timothy's World Coffee",
        'location': {'lat': 43.65123,
         'lng': -79.368457,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.65123,
           'lng': -79.368457}],
         'distance': 1271,
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['Toronto ON', 'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False},
       {'id': '5ae32b6412c8f0002c2b03e7',
        'name': 'Super Jet International Coffee Shop',
        'location': {'address': '267 College St.',
         'crossStreet': 'Spadina Ave',
         'lat': 43.657971,
         'lng': -79.399795,
         'labeledLatLngs': [{'label': 'display',
           'lat': 43.657971,
           'lng': -79.399795}],
         'distance': 1371,
         'postalCode': 'M5T 1R5',
         'cc': 'CA',
         'city': 'Toronto',
         'state': 'ON',
         'country': 'Canada',
         'formattedAddress': ['267 College St. (Spadina Ave)',
          'Toronto ON M5T 1R5',
          'Canada']},
        'categories': [{'id': '4bf58dd8d48988d1e0931735',
          'name': 'Coffee Shop',
          'pluralName': 'Coffee Shops',
          'shortName': 'Coffee Shop',
          'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
           'suffix': '.png'},
          'primary': True}],
        'referralId': 'v-1616348892',
        'hasPerk': False}]}}



**Converting the list of Coffee Shops into DataFrame**


```python
# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a dataframe
dataframe = json_normalize(venues)
dataframe
```

    /opt/conda/envs/Python-3.7-main/lib/python3.7/site-packages/ipykernel/__main__.py:5: FutureWarning: pandas.io.json.json_normalize is deprecated, use pandas.json_normalize instead





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>categories</th>
      <th>referralId</th>
      <th>hasPerk</th>
      <th>location.address</th>
      <th>location.crossStreet</th>
      <th>location.lat</th>
      <th>location.lng</th>
      <th>location.labeledLatLngs</th>
      <th>location.distance</th>
      <th>location.postalCode</th>
      <th>location.cc</th>
      <th>location.neighborhood</th>
      <th>location.city</th>
      <th>location.state</th>
      <th>location.country</th>
      <th>location.formattedAddress</th>
      <th>venuePage.id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59f784dd28122f14f9d5d63d</td>
      <td>HotBlack Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>245 Queen Street West</td>
      <td>at St Patrick St</td>
      <td>43.650364</td>
      <td>-79.388669</td>
      <td>[{'label': 'display', 'lat': 43.65036434800487...</td>
      <td>515</td>
      <td>M5V 1Z4</td>
      <td>CA</td>
      <td>Entertainment District</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[245 Queen Street West (at St Patrick St), Tor...</td>
      <td>463001529</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4b44fc77f964a520cc0026e3</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>427 University Avenue</td>
      <td>NaN</td>
      <td>43.654053</td>
      <td>-79.388090</td>
      <td>[{'label': 'display', 'lat': 43.65405317976302...</td>
      <td>340</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[427 University Avenue, Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4b0aaa8ef964a520272623e3</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>483 Bay St, Bell Trinity Square</td>
      <td>Bell Trinity Square</td>
      <td>43.653436</td>
      <td>-79.382314</td>
      <td>[{'label': 'display', 'lat': 43.653436, 'lng':...</td>
      <td>130</td>
      <td>M5G 2C9</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[483 Bay St, Bell Trinity Square (Bell Trinity...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4fb13c20e4b011e6f93513c0</td>
      <td>Balzac's Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>122 Bond Street</td>
      <td>at Gould St.</td>
      <td>43.657854</td>
      <td>-79.379200</td>
      <td>[{'label': 'display', 'lat': 43.65785440672277...</td>
      <td>618</td>
      <td>M5B 1X8</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[122 Bond Street (at Gould St.), Toronto ON M5...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4baa9f6cf964a520817a3ae3</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>401 Bay St.</td>
      <td>at Richmond St. W</td>
      <td>43.652135</td>
      <td>-79.381172</td>
      <td>[{'label': 'display', 'lat': 43.65213455850074...</td>
      <td>268</td>
      <td>M5H 2Y4</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[401 Bay St. (at Richmond St. W), Toronto ON M...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>53e8acc4498ee294fb100183</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>425 University Ave</td>
      <td>Dundas</td>
      <td>43.654270</td>
      <td>-79.387448</td>
      <td>[{'label': 'display', 'lat': 43.65427, 'lng': ...</td>
      <td>296</td>
      <td>M5G 1T6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[425 University Ave (Dundas), Toronto ON M5G 1...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4baa31def964a52037523ae3</td>
      <td>Coffee office</td>
      <td>[]</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>350 Bay St - 7th Floor</td>
      <td>NaN</td>
      <td>43.649498</td>
      <td>-79.386479</td>
      <td>[{'label': 'display', 'lat': 43.649498, 'lng':...</td>
      <td>488</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[350 Bay St - 7th Floor, Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4fff1f96e4b042ae8acddca5</td>
      <td>Fahrenheit Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>120 Lombard St</td>
      <td>at Jarvis St</td>
      <td>43.652384</td>
      <td>-79.372719</td>
      <td>[{'label': 'display', 'lat': 43.65238358726612...</td>
      <td>911</td>
      <td>M5C 3H5</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[120 Lombard St (at Jarvis St), Toronto ON M5C...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4ec514ec9911232436e364af</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>Yonge</td>
      <td>Dundas</td>
      <td>43.656700</td>
      <td>-79.379941</td>
      <td>[{'label': 'display', 'lat': 43.65669995833159...</td>
      <td>481</td>
      <td>M5B 2G9</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[Yonge (Dundas), Toronto ON M5B 2G9, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4fccaa8fe4b05a98df3d9417</td>
      <td>Sam James Coffee Bar (SJCB)</td>
      <td>[{'id': '4bf58dd8d48988d16d941735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>150 King St. W</td>
      <td>in the PATH</td>
      <td>43.647881</td>
      <td>-79.384332</td>
      <td>[{'label': 'display', 'lat': 43.64788137014028...</td>
      <td>624</td>
      <td>M5H 4B6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[150 King St. W (in the PATH), Toronto ON M5H ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>536fd522498e09c6690800e2</td>
      <td>Balzac's Coffee</td>
      <td>[{'id': '4bf58dd8d48988d16d941735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>10 Market Street</td>
      <td>btwn The Esplanade &amp; Front St. E.</td>
      <td>43.648457</td>
      <td>-79.371790</td>
      <td>[{'label': 'display', 'lat': 43.64845650131932...</td>
      <td>1126</td>
      <td>M5E 1M6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[10 Market Street (btwn The Esplanade &amp; Front ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4ad79243f964a5204c0c21e3</td>
      <td>Jetfuel Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>519 Parliament St.</td>
      <td>btwn Carlton &amp; Winchester</td>
      <td>43.665295</td>
      <td>-79.368335</td>
      <td>[{'label': 'display', 'lat': 43.66529519392083...</td>
      <td>1818</td>
      <td>M4X 1P3</td>
      <td>CA</td>
      <td>Cabbagetown</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[519 Parliament St. (btwn Carlton &amp; Winchester...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>563d2f2dcd10bcf27ae37c3b</td>
      <td>Pilot Coffee Roasters</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>65 Front St W</td>
      <td>btwn Bay St &amp; York St</td>
      <td>43.645018</td>
      <td>-79.380415</td>
      <td>[{'label': 'display', 'lat': 43.64501814464698...</td>
      <td>983</td>
      <td>M5J 1E6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[65 Front St W (btwn Bay St &amp; York St), Toront...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5c86b682da2e00002cf95781</td>
      <td>Second Cup Coffee Co. featuring Pinkberry Froz...</td>
      <td>[{'id': '4bf58dd8d48988d16d941735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>600 University Avenue, Room #202</td>
      <td>NaN</td>
      <td>43.657473</td>
      <td>-79.390637</td>
      <td>[{'label': 'display', 'lat': 43.657473, 'lng':...</td>
      <td>699</td>
      <td>M5G 1X5</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[600 University Avenue, Room #202, Toronto ON ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4d261e1e3c84b1f78bf70847</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>30 Adelaide St E</td>
      <td>NaN</td>
      <td>43.650948</td>
      <td>-79.376825</td>
      <td>[{'label': 'display', 'lat': 43.650948, 'lng':...</td>
      <td>638</td>
      <td>M5C 3G8</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[30 Adelaide St E, Toronto ON M5C 3G8, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>569e7814498e1a7f3e01bfe4</td>
      <td>Rooster Coffee House</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>568 Jarvis St</td>
      <td>At Charles St E</td>
      <td>43.669654</td>
      <td>-79.379871</td>
      <td>[{'label': 'display', 'lat': 43.66965378571954...</td>
      <td>1829</td>
      <td>M4Y 1N6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[568 Jarvis St (At Charles St E), Toronto ON M...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4bce5e21cc8cd13a7359c4cf</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>444 Yonge St</td>
      <td>in College Park</td>
      <td>43.660467</td>
      <td>-79.384654</td>
      <td>[{'label': 'display', 'lat': 43.66046739684086...</td>
      <td>779</td>
      <td>M5B 2H4</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[444 Yonge St (in College Park), Toronto ON M5...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4d8a7096d85f3704d05afedb</td>
      <td>T.A.N. Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>37 Baldwin St</td>
      <td>Henry St</td>
      <td>43.656029</td>
      <td>-79.393534</td>
      <td>[{'label': 'display', 'lat': 43.65602860741956...</td>
      <td>823</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[37 Baldwin St (Henry St), Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4ba37627f964a520263f38e3</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>66 Wellington Street West</td>
      <td>TD Center Concourse</td>
      <td>43.647130</td>
      <td>-79.380776</td>
      <td>[{'label': 'display', 'lat': 43.64713049355658...</td>
      <td>751</td>
      <td>M5K 1A1</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[66 Wellington Street West (TD Center Concours...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>51438b33e4b0a40e33fe5e77</td>
      <td>Jimmy's Coffee</td>
      <td>[{'id': '4bf58dd8d48988d16d941735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>191 Baldwin St</td>
      <td>Kensington Market</td>
      <td>43.654493</td>
      <td>-79.401311</td>
      <td>[{'label': 'display', 'lat': 43.65449315540114...</td>
      <td>1404</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Kensington Market</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[191 Baldwin St (Kensington Market), Toronto O...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5553954a498e8e11bc49ecf2</td>
      <td>Sam James Coffee Bar (SJCB)</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>15 Toronto Street</td>
      <td>NaN</td>
      <td>43.650319</td>
      <td>-79.376217</td>
      <td>[{'label': 'display', 'lat': 43.65031871629752...</td>
      <td>714</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[15 Toronto Street, Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4b156e98f964a520cbac23e3</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>801 Bay St</td>
      <td>at College St</td>
      <td>43.660714</td>
      <td>-79.385491</td>
      <td>[{'label': 'display', 'lat': 43.66071353922905...</td>
      <td>814</td>
      <td>M5S 1Y9</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[801 Bay St (at College St), Toronto ON M5S 1Y...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>557f05ca498ec78ac7b29315</td>
      <td>Balzac's Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>7 Station St</td>
      <td>at SkyWalk</td>
      <td>43.644373</td>
      <td>-79.383065</td>
      <td>[{'label': 'display', 'lat': 43.64437258414836...</td>
      <td>1016</td>
      <td>M5J 1C3</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[7 Station St (at SkyWalk), Toronto ON M5J 1C3...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5c86b6b9cb3fd2002cd9c9a2</td>
      <td>Second Cup Coffee Co. featuring Pinkberry Froz...</td>
      <td>[{'id': '4bf58dd8d48988d16d941735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>179 College Street</td>
      <td>NaN</td>
      <td>43.658872</td>
      <td>-79.394158</td>
      <td>[{'label': 'display', 'lat': 43.65887244042741...</td>
      <td>1018</td>
      <td>M5T 1P7</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[179 College Street, Toronto ON M5T 1P7, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>584593c2ebf0284fe7b103cb</td>
      <td>Fahrenheit Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>529 Richmond St W</td>
      <td>NaN</td>
      <td>43.647037</td>
      <td>-79.400876</td>
      <td>[{'label': 'display', 'lat': 43.64703669923361...</td>
      <td>1541</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Fashion District</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[529 Richmond St W, Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4b9e7808f964a52091e636e3</td>
      <td>Second Cup Coffee Co.</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>200 Front St W</td>
      <td>in Simcoe Place</td>
      <td>43.645009</td>
      <td>-79.385812</td>
      <td>[{'label': 'display', 'lat': 43.64500941272016...</td>
      <td>955</td>
      <td>M5V 3K2</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[200 Front St W (in Simcoe Place), Toronto ON ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>4adb5a00f964a5204c2621e3</td>
      <td>I Deal Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>84 Nassau Street</td>
      <td>NaN</td>
      <td>43.655058</td>
      <td>-79.403254</td>
      <td>[{'label': 'display', 'lat': 43.65505778213131...</td>
      <td>1565</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[84 Nassau Street, Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4b758da6f964a520cb132ee3</td>
      <td>Luba’s Coffee &amp; Tea Boutique</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.649053</td>
      <td>-79.371981</td>
      <td>[{'label': 'display', 'lat': 43.6490525948081,...</td>
      <td>1081</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4c3de34c7d002d7fe460b018</td>
      <td>Timothy's World Coffee</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.651230</td>
      <td>-79.368457</td>
      <td>[{'label': 'display', 'lat': 43.65123, 'lng': ...</td>
      <td>1271</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[Toronto ON, Canada]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5ae32b6412c8f0002c2b03e7</td>
      <td>Super Jet International Coffee Shop</td>
      <td>[{'id': '4bf58dd8d48988d1e0931735', 'name': 'C...</td>
      <td>v-1616348892</td>
      <td>False</td>
      <td>267 College St.</td>
      <td>Spadina Ave</td>
      <td>43.657971</td>
      <td>-79.399795</td>
      <td>[{'label': 'display', 'lat': 43.657971, 'lng':...</td>
      <td>1371</td>
      <td>M5T 1R5</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[267 College St. (Spadina Ave), Toronto ON M5T...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Filtering the result set**


```python
# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>address</th>
      <th>crossStreet</th>
      <th>lat</th>
      <th>lng</th>
      <th>labeledLatLngs</th>
      <th>distance</th>
      <th>postalCode</th>
      <th>cc</th>
      <th>neighborhood</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>formattedAddress</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HotBlack Coffee</td>
      <td>Coffee Shop</td>
      <td>245 Queen Street West</td>
      <td>at St Patrick St</td>
      <td>43.650364</td>
      <td>-79.388669</td>
      <td>[{'label': 'display', 'lat': 43.65036434800487...</td>
      <td>515</td>
      <td>M5V 1Z4</td>
      <td>CA</td>
      <td>Entertainment District</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[245 Queen Street West (at St Patrick St), Tor...</td>
      <td>59f784dd28122f14f9d5d63d</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>427 University Avenue</td>
      <td>NaN</td>
      <td>43.654053</td>
      <td>-79.388090</td>
      <td>[{'label': 'display', 'lat': 43.65405317976302...</td>
      <td>340</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[427 University Avenue, Toronto ON, Canada]</td>
      <td>4b44fc77f964a520cc0026e3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>483 Bay St, Bell Trinity Square</td>
      <td>Bell Trinity Square</td>
      <td>43.653436</td>
      <td>-79.382314</td>
      <td>[{'label': 'display', 'lat': 43.653436, 'lng':...</td>
      <td>130</td>
      <td>M5G 2C9</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[483 Bay St, Bell Trinity Square (Bell Trinity...</td>
      <td>4b0aaa8ef964a520272623e3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Balzac's Coffee</td>
      <td>Coffee Shop</td>
      <td>122 Bond Street</td>
      <td>at Gould St.</td>
      <td>43.657854</td>
      <td>-79.379200</td>
      <td>[{'label': 'display', 'lat': 43.65785440672277...</td>
      <td>618</td>
      <td>M5B 1X8</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[122 Bond Street (at Gould St.), Toronto ON M5...</td>
      <td>4fb13c20e4b011e6f93513c0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>401 Bay St.</td>
      <td>at Richmond St. W</td>
      <td>43.652135</td>
      <td>-79.381172</td>
      <td>[{'label': 'display', 'lat': 43.65213455850074...</td>
      <td>268</td>
      <td>M5H 2Y4</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[401 Bay St. (at Richmond St. W), Toronto ON M...</td>
      <td>4baa9f6cf964a520817a3ae3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>425 University Ave</td>
      <td>Dundas</td>
      <td>43.654270</td>
      <td>-79.387448</td>
      <td>[{'label': 'display', 'lat': 43.65427, 'lng': ...</td>
      <td>296</td>
      <td>M5G 1T6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[425 University Ave (Dundas), Toronto ON M5G 1...</td>
      <td>53e8acc4498ee294fb100183</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Coffee office</td>
      <td>None</td>
      <td>350 Bay St - 7th Floor</td>
      <td>NaN</td>
      <td>43.649498</td>
      <td>-79.386479</td>
      <td>[{'label': 'display', 'lat': 43.649498, 'lng':...</td>
      <td>488</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[350 Bay St - 7th Floor, Toronto ON, Canada]</td>
      <td>4baa31def964a52037523ae3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fahrenheit Coffee</td>
      <td>Coffee Shop</td>
      <td>120 Lombard St</td>
      <td>at Jarvis St</td>
      <td>43.652384</td>
      <td>-79.372719</td>
      <td>[{'label': 'display', 'lat': 43.65238358726612...</td>
      <td>911</td>
      <td>M5C 3H5</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[120 Lombard St (at Jarvis St), Toronto ON M5C...</td>
      <td>4fff1f96e4b042ae8acddca5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>Yonge</td>
      <td>Dundas</td>
      <td>43.656700</td>
      <td>-79.379941</td>
      <td>[{'label': 'display', 'lat': 43.65669995833159...</td>
      <td>481</td>
      <td>M5B 2G9</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[Yonge (Dundas), Toronto ON M5B 2G9, Canada]</td>
      <td>4ec514ec9911232436e364af</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sam James Coffee Bar (SJCB)</td>
      <td>Café</td>
      <td>150 King St. W</td>
      <td>in the PATH</td>
      <td>43.647881</td>
      <td>-79.384332</td>
      <td>[{'label': 'display', 'lat': 43.64788137014028...</td>
      <td>624</td>
      <td>M5H 4B6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[150 King St. W (in the PATH), Toronto ON M5H ...</td>
      <td>4fccaa8fe4b05a98df3d9417</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Balzac's Coffee</td>
      <td>Café</td>
      <td>10 Market Street</td>
      <td>btwn The Esplanade &amp; Front St. E.</td>
      <td>43.648457</td>
      <td>-79.371790</td>
      <td>[{'label': 'display', 'lat': 43.64845650131932...</td>
      <td>1126</td>
      <td>M5E 1M6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[10 Market Street (btwn The Esplanade &amp; Front ...</td>
      <td>536fd522498e09c6690800e2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Jetfuel Coffee</td>
      <td>Coffee Shop</td>
      <td>519 Parliament St.</td>
      <td>btwn Carlton &amp; Winchester</td>
      <td>43.665295</td>
      <td>-79.368335</td>
      <td>[{'label': 'display', 'lat': 43.66529519392083...</td>
      <td>1818</td>
      <td>M4X 1P3</td>
      <td>CA</td>
      <td>Cabbagetown</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[519 Parliament St. (btwn Carlton &amp; Winchester...</td>
      <td>4ad79243f964a5204c0c21e3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Pilot Coffee Roasters</td>
      <td>Coffee Shop</td>
      <td>65 Front St W</td>
      <td>btwn Bay St &amp; York St</td>
      <td>43.645018</td>
      <td>-79.380415</td>
      <td>[{'label': 'display', 'lat': 43.64501814464698...</td>
      <td>983</td>
      <td>M5J 1E6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[65 Front St W (btwn Bay St &amp; York St), Toront...</td>
      <td>563d2f2dcd10bcf27ae37c3b</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Second Cup Coffee Co. featuring Pinkberry Froz...</td>
      <td>Café</td>
      <td>600 University Avenue, Room #202</td>
      <td>NaN</td>
      <td>43.657473</td>
      <td>-79.390637</td>
      <td>[{'label': 'display', 'lat': 43.657473, 'lng':...</td>
      <td>699</td>
      <td>M5G 1X5</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[600 University Avenue, Room #202, Toronto ON ...</td>
      <td>5c86b682da2e00002cf95781</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>30 Adelaide St E</td>
      <td>NaN</td>
      <td>43.650948</td>
      <td>-79.376825</td>
      <td>[{'label': 'display', 'lat': 43.650948, 'lng':...</td>
      <td>638</td>
      <td>M5C 3G8</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[30 Adelaide St E, Toronto ON M5C 3G8, Canada]</td>
      <td>4d261e1e3c84b1f78bf70847</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Rooster Coffee House</td>
      <td>Coffee Shop</td>
      <td>568 Jarvis St</td>
      <td>At Charles St E</td>
      <td>43.669654</td>
      <td>-79.379871</td>
      <td>[{'label': 'display', 'lat': 43.66965378571954...</td>
      <td>1829</td>
      <td>M4Y 1N6</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[568 Jarvis St (At Charles St E), Toronto ON M...</td>
      <td>569e7814498e1a7f3e01bfe4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>444 Yonge St</td>
      <td>in College Park</td>
      <td>43.660467</td>
      <td>-79.384654</td>
      <td>[{'label': 'display', 'lat': 43.66046739684086...</td>
      <td>779</td>
      <td>M5B 2H4</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[444 Yonge St (in College Park), Toronto ON M5...</td>
      <td>4bce5e21cc8cd13a7359c4cf</td>
    </tr>
    <tr>
      <th>17</th>
      <td>T.A.N. Coffee</td>
      <td>Coffee Shop</td>
      <td>37 Baldwin St</td>
      <td>Henry St</td>
      <td>43.656029</td>
      <td>-79.393534</td>
      <td>[{'label': 'display', 'lat': 43.65602860741956...</td>
      <td>823</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[37 Baldwin St (Henry St), Toronto ON, Canada]</td>
      <td>4d8a7096d85f3704d05afedb</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>66 Wellington Street West</td>
      <td>TD Center Concourse</td>
      <td>43.647130</td>
      <td>-79.380776</td>
      <td>[{'label': 'display', 'lat': 43.64713049355658...</td>
      <td>751</td>
      <td>M5K 1A1</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[66 Wellington Street West (TD Center Concours...</td>
      <td>4ba37627f964a520263f38e3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Jimmy's Coffee</td>
      <td>Café</td>
      <td>191 Baldwin St</td>
      <td>Kensington Market</td>
      <td>43.654493</td>
      <td>-79.401311</td>
      <td>[{'label': 'display', 'lat': 43.65449315540114...</td>
      <td>1404</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Kensington Market</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[191 Baldwin St (Kensington Market), Toronto O...</td>
      <td>51438b33e4b0a40e33fe5e77</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sam James Coffee Bar (SJCB)</td>
      <td>Coffee Shop</td>
      <td>15 Toronto Street</td>
      <td>NaN</td>
      <td>43.650319</td>
      <td>-79.376217</td>
      <td>[{'label': 'display', 'lat': 43.65031871629752...</td>
      <td>714</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[15 Toronto Street, Toronto ON, Canada]</td>
      <td>5553954a498e8e11bc49ecf2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>801 Bay St</td>
      <td>at College St</td>
      <td>43.660714</td>
      <td>-79.385491</td>
      <td>[{'label': 'display', 'lat': 43.66071353922905...</td>
      <td>814</td>
      <td>M5S 1Y9</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[801 Bay St (at College St), Toronto ON M5S 1Y...</td>
      <td>4b156e98f964a520cbac23e3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Balzac's Coffee</td>
      <td>Coffee Shop</td>
      <td>7 Station St</td>
      <td>at SkyWalk</td>
      <td>43.644373</td>
      <td>-79.383065</td>
      <td>[{'label': 'display', 'lat': 43.64437258414836...</td>
      <td>1016</td>
      <td>M5J 1C3</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[7 Station St (at SkyWalk), Toronto ON M5J 1C3...</td>
      <td>557f05ca498ec78ac7b29315</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Second Cup Coffee Co. featuring Pinkberry Froz...</td>
      <td>Café</td>
      <td>179 College Street</td>
      <td>NaN</td>
      <td>43.658872</td>
      <td>-79.394158</td>
      <td>[{'label': 'display', 'lat': 43.65887244042741...</td>
      <td>1018</td>
      <td>M5T 1P7</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[179 College Street, Toronto ON M5T 1P7, Canada]</td>
      <td>5c86b6b9cb3fd2002cd9c9a2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Fahrenheit Coffee</td>
      <td>Coffee Shop</td>
      <td>529 Richmond St W</td>
      <td>NaN</td>
      <td>43.647037</td>
      <td>-79.400876</td>
      <td>[{'label': 'display', 'lat': 43.64703669923361...</td>
      <td>1541</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Fashion District</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[529 Richmond St W, Toronto ON, Canada]</td>
      <td>584593c2ebf0284fe7b103cb</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Second Cup Coffee Co.</td>
      <td>Coffee Shop</td>
      <td>200 Front St W</td>
      <td>in Simcoe Place</td>
      <td>43.645009</td>
      <td>-79.385812</td>
      <td>[{'label': 'display', 'lat': 43.64500941272016...</td>
      <td>955</td>
      <td>M5V 3K2</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[200 Front St W (in Simcoe Place), Toronto ON ...</td>
      <td>4b9e7808f964a52091e636e3</td>
    </tr>
    <tr>
      <th>26</th>
      <td>I Deal Coffee</td>
      <td>Coffee Shop</td>
      <td>84 Nassau Street</td>
      <td>NaN</td>
      <td>43.655058</td>
      <td>-79.403254</td>
      <td>[{'label': 'display', 'lat': 43.65505778213131...</td>
      <td>1565</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[84 Nassau Street, Toronto ON, Canada]</td>
      <td>4adb5a00f964a5204c2621e3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Luba’s Coffee &amp; Tea Boutique</td>
      <td>Coffee Shop</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.649053</td>
      <td>-79.371981</td>
      <td>[{'label': 'display', 'lat': 43.6490525948081,...</td>
      <td>1081</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[Toronto ON, Canada]</td>
      <td>4b758da6f964a520cb132ee3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Timothy's World Coffee</td>
      <td>Coffee Shop</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.651230</td>
      <td>-79.368457</td>
      <td>[{'label': 'display', 'lat': 43.65123, 'lng': ...</td>
      <td>1271</td>
      <td>NaN</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[Toronto ON, Canada]</td>
      <td>4c3de34c7d002d7fe460b018</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Super Jet International Coffee Shop</td>
      <td>Coffee Shop</td>
      <td>267 College St.</td>
      <td>Spadina Ave</td>
      <td>43.657971</td>
      <td>-79.399795</td>
      <td>[{'label': 'display', 'lat': 43.657971, 'lng':...</td>
      <td>1371</td>
      <td>M5T 1R5</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>Canada</td>
      <td>[267 College St. (Spadina Ave), Toronto ON M5T...</td>
      <td>5ae32b6412c8f0002c2b03e7</td>
    </tr>
  </tbody>
</table>
</div>



**Getting the list of coffee Shops**


```python
dataframe_filtered.name
```




    0                                       HotBlack Coffee
    1                                Timothy's World Coffee
    2                                Timothy's World Coffee
    3                                       Balzac's Coffee
    4                                Timothy's World Coffee
    5                                Timothy's World Coffee
    6                                         Coffee office
    7                                     Fahrenheit Coffee
    8                                Timothy's World Coffee
    9                           Sam James Coffee Bar (SJCB)
    10                                      Balzac's Coffee
    11                                       Jetfuel Coffee
    12                                Pilot Coffee Roasters
    13    Second Cup Coffee Co. featuring Pinkberry Froz...
    14                               Timothy's World Coffee
    15                                 Rooster Coffee House
    16                               Timothy's World Coffee
    17                                        T.A.N. Coffee
    18                               Timothy's World Coffee
    19                                       Jimmy's Coffee
    20                          Sam James Coffee Bar (SJCB)
    21                               Timothy's World Coffee
    22                                      Balzac's Coffee
    23    Second Cup Coffee Co. featuring Pinkberry Froz...
    24                                    Fahrenheit Coffee
    25                                Second Cup Coffee Co.
    26                                        I Deal Coffee
    27                         Luba’s Coffee & Tea Boutique
    28                               Timothy's World Coffee
    29                  Super Jet International Coffee Shop
    Name: name, dtype: object




```python
venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) # generate map centred around the Conrad Hotel

# add a red circle marker to represent the Conrad Hotel
folium.CircleMarker(
    [latitude, longitude],
    radius=10,
    color='red',
    popup='Toronto,Canada',
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

# add the Italian restaurants as blue circle markers
for lat, lng, label in zip(dataframe_filtered.lat, dataframe_filtered.lng, dataframe_filtered.categories):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map
venues_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%3Cscript%3EL_PREFER_CANVAS%20%3D%20false%3B%20L_NO_TOUCH%20%3D%20false%3B%20L_DISABLE_3D%20%3D%20false%3B%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.2.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css%22/%3E%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%20%23map_1f0f4b4ce93e4efbaf0e37354abe5653%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%20%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%20%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_1f0f4b4ce93e4efbaf0e37354abe5653%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20bounds%20%3D%20null%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_1f0f4b4ce93e4efbaf0e37354abe5653%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27map_1f0f4b4ce93e4efbaf0e37354abe5653%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7Bcenter%3A%20%5B43.6534817%2C-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2013%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20maxBounds%3A%20bounds%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20layers%3A%20%5B%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20worldCopyJump%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_a43625065b294e6aba715efc758d2b5c%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%27https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%27%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22attribution%22%3A%20null%2C%0A%20%20%22detectRetina%22%3A%20false%2C%0A%20%20%22maxZoom%22%3A%2018%2C%0A%20%20%22minZoom%22%3A%201%2C%0A%20%20%22noWrap%22%3A%20false%2C%0A%20%20%22subdomains%22%3A%20%22abc%22%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a186d7b2fa2c4246b995e06c505390d5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6534817%2C-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22red%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22red%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%2010%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_291c99b53dd44821bafadca6f6595dda%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_435bec8509a34f1d9b403ad853235213%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_435bec8509a34f1d9b403ad853235213%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EToronto%2CCanada%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_291c99b53dd44821bafadca6f6595dda.setContent%28html_435bec8509a34f1d9b403ad853235213%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_a186d7b2fa2c4246b995e06c505390d5.bindPopup%28popup_291c99b53dd44821bafadca6f6595dda%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_829e03b7123b4020842223c7149c8eaf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65036434800487%2C-79.38866907575726%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_71a21c4a77884aad9a11374811b19549%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b9ed30b9e9064239a4ab1c81dfbe51f1%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b9ed30b9e9064239a4ab1c81dfbe51f1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_71a21c4a77884aad9a11374811b19549.setContent%28html_b9ed30b9e9064239a4ab1c81dfbe51f1%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_829e03b7123b4020842223c7149c8eaf.bindPopup%28popup_71a21c4a77884aad9a11374811b19549%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_904571688216431392f1b1f84a79aa9d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65405317976302%2C-79.38808999785911%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_5a445fb2ff704470894a9fd33102da91%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d7a6aa5820bf43cc985e1f20efbb6767%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d7a6aa5820bf43cc985e1f20efbb6767%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_5a445fb2ff704470894a9fd33102da91.setContent%28html_d7a6aa5820bf43cc985e1f20efbb6767%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_904571688216431392f1b1f84a79aa9d.bindPopup%28popup_5a445fb2ff704470894a9fd33102da91%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2e70ccb6be4e4d98a5fc75417e9aed11%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.653436%2C-79.382314%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_9b919cb779d6428db9b5e826c5baa9ea%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_35bceb1f40ec4d829b5757163463a806%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_35bceb1f40ec4d829b5757163463a806%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_9b919cb779d6428db9b5e826c5baa9ea.setContent%28html_35bceb1f40ec4d829b5757163463a806%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_2e70ccb6be4e4d98a5fc75417e9aed11.bindPopup%28popup_9b919cb779d6428db9b5e826c5baa9ea%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cd25f84806654bf1858a779fd39d788f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65785440672277%2C-79.37919981155157%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_55fa47569c2240749e71d1d0ce1c39d9%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_319514af704b4cfd8dc6f2b20362acf0%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_319514af704b4cfd8dc6f2b20362acf0%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_55fa47569c2240749e71d1d0ce1c39d9.setContent%28html_319514af704b4cfd8dc6f2b20362acf0%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cd25f84806654bf1858a779fd39d788f.bindPopup%28popup_55fa47569c2240749e71d1d0ce1c39d9%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_daba3b6ef7ad415e8bc70fe2119fe35a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65213455850074%2C-79.38117224696582%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8fb5b5c392354acda1884c8df9b34434%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b599c1cc779b4fdaad12e759b9a5ecdf%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b599c1cc779b4fdaad12e759b9a5ecdf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8fb5b5c392354acda1884c8df9b34434.setContent%28html_b599c1cc779b4fdaad12e759b9a5ecdf%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_daba3b6ef7ad415e8bc70fe2119fe35a.bindPopup%28popup_8fb5b5c392354acda1884c8df9b34434%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d1360c36583441c28f2d7bfec09b078b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65427%2C-79.387448%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_690ba57a4ce748a68b90404d259eb560%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d824be8553714ddcb739f9561df66437%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d824be8553714ddcb739f9561df66437%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_690ba57a4ce748a68b90404d259eb560.setContent%28html_d824be8553714ddcb739f9561df66437%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_d1360c36583441c28f2d7bfec09b078b.bindPopup%28popup_690ba57a4ce748a68b90404d259eb560%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4b937984579b4f92b39b8d210fc6af04%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.649498%2C-79.386479%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5722a261e8e2487f8244cb81ef4808cc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65238358726612%2C-79.37271903848271%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d444cc0add0d44429533f7854534fc8a%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1ac7d8e77d0840ca91bdc6524381e546%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1ac7d8e77d0840ca91bdc6524381e546%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d444cc0add0d44429533f7854534fc8a.setContent%28html_1ac7d8e77d0840ca91bdc6524381e546%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_5722a261e8e2487f8244cb81ef4808cc.bindPopup%28popup_d444cc0add0d44429533f7854534fc8a%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2e540766d7074104b5ec0dac077ef8bb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65669995833159%2C-79.37994058195848%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_a3e5151554b342e193a9135af233fe6e%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1e2beb10ec76400487064191b6d496b6%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_1e2beb10ec76400487064191b6d496b6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_a3e5151554b342e193a9135af233fe6e.setContent%28html_1e2beb10ec76400487064191b6d496b6%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_2e540766d7074104b5ec0dac077ef8bb.bindPopup%28popup_a3e5151554b342e193a9135af233fe6e%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_552e8e47860746b7a75d06f30fe89d79%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64788137014028%2C-79.38433152836829%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_fe5ca9d4b8454aa2a18e3a1b729b0186%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_804a98a12c6f4db0afdeb922f3228c78%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_804a98a12c6f4db0afdeb922f3228c78%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaf%C3%A9%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_fe5ca9d4b8454aa2a18e3a1b729b0186.setContent%28html_804a98a12c6f4db0afdeb922f3228c78%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_552e8e47860746b7a75d06f30fe89d79.bindPopup%28popup_fe5ca9d4b8454aa2a18e3a1b729b0186%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b14fed23e99f43ac937a5c14be0b7684%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64845650131932%2C-79.37178993724407%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_63e9d640735245719a36823c240239ff%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_174d43fb1f2a4eb6ab06e3920590c2b3%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_174d43fb1f2a4eb6ab06e3920590c2b3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaf%C3%A9%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_63e9d640735245719a36823c240239ff.setContent%28html_174d43fb1f2a4eb6ab06e3920590c2b3%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_b14fed23e99f43ac937a5c14be0b7684.bindPopup%28popup_63e9d640735245719a36823c240239ff%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9bec6c7316834fc9a7af9243297cb658%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.66529519392083%2C-79.3683345416816%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d1448d92dd07403bb25934a46b225abb%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3397f8f8682240bd8ac0078ec3c345b1%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_3397f8f8682240bd8ac0078ec3c345b1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d1448d92dd07403bb25934a46b225abb.setContent%28html_3397f8f8682240bd8ac0078ec3c345b1%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9bec6c7316834fc9a7af9243297cb658.bindPopup%28popup_d1448d92dd07403bb25934a46b225abb%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_468bd2a3c7b7471ea7392581fa1a9354%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64501814464698%2C-79.3804150931199%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_dcf6eaf0840048858135a923aeb50054%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f1b99370356a48d2a015192525fa620d%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_f1b99370356a48d2a015192525fa620d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_dcf6eaf0840048858135a923aeb50054.setContent%28html_f1b99370356a48d2a015192525fa620d%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_468bd2a3c7b7471ea7392581fa1a9354.bindPopup%28popup_dcf6eaf0840048858135a923aeb50054%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_05144fae5d4e4466928b2024f176be9a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.657473%2C-79.390637%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_902a8b6bac534565b2336890ae5e36c6%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b9a6a78b12ae4faf95cfa2fe904f6480%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b9a6a78b12ae4faf95cfa2fe904f6480%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaf%C3%A9%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_902a8b6bac534565b2336890ae5e36c6.setContent%28html_b9a6a78b12ae4faf95cfa2fe904f6480%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_05144fae5d4e4466928b2024f176be9a.bindPopup%28popup_902a8b6bac534565b2336890ae5e36c6%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6e1e66a371c24758932b19b8f056b3fa%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.650948%2C-79.376825%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_1dd8559b277d4b5c979ff408e027b210%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_95ab4fe1d13d41cb97ec0c29fe0b1e00%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_95ab4fe1d13d41cb97ec0c29fe0b1e00%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_1dd8559b277d4b5c979ff408e027b210.setContent%28html_95ab4fe1d13d41cb97ec0c29fe0b1e00%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_6e1e66a371c24758932b19b8f056b3fa.bindPopup%28popup_1dd8559b277d4b5c979ff408e027b210%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0bfbcb86be9346669d9f04da5c115016%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.66965378571954%2C-79.379870566686%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_3a3236542b8e48a6a878369d86064085%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d615f8f9339642ce876778dfdc1a5713%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_d615f8f9339642ce876778dfdc1a5713%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_3a3236542b8e48a6a878369d86064085.setContent%28html_d615f8f9339642ce876778dfdc1a5713%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_0bfbcb86be9346669d9f04da5c115016.bindPopup%28popup_3a3236542b8e48a6a878369d86064085%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3b150d9032cf484ba697a14293196d91%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.66046739684086%2C-79.38465356826782%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_e6cf57feab5d41ed942c121c701da69d%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5a2c0702e52047208e88f41c7e938ba4%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_5a2c0702e52047208e88f41c7e938ba4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_e6cf57feab5d41ed942c121c701da69d.setContent%28html_5a2c0702e52047208e88f41c7e938ba4%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_3b150d9032cf484ba697a14293196d91.bindPopup%28popup_e6cf57feab5d41ed942c121c701da69d%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9ed9eb33d6c643f2b19265adb3a6c94b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65602860741956%2C-79.39353447011945%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_8285c7b044a542f8bea1b48f38792d23%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9a41577f5c304081bacd0e8590db2079%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9a41577f5c304081bacd0e8590db2079%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_8285c7b044a542f8bea1b48f38792d23.setContent%28html_9a41577f5c304081bacd0e8590db2079%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9ed9eb33d6c643f2b19265adb3a6c94b.bindPopup%28popup_8285c7b044a542f8bea1b48f38792d23%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9066a1c4093a48dfb97505a0b6c10611%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64713049355658%2C-79.38077635642868%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_464e83648f374bfa8b549ece71b3a1db%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_29fcb3a42002430e94e7d7c6ae7af78a%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_29fcb3a42002430e94e7d7c6ae7af78a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_464e83648f374bfa8b549ece71b3a1db.setContent%28html_29fcb3a42002430e94e7d7c6ae7af78a%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_9066a1c4093a48dfb97505a0b6c10611.bindPopup%28popup_464e83648f374bfa8b549ece71b3a1db%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e5230105d6ed40f08f5425cd89dde7f1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65449315540114%2C-79.40131090393002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d73028a0b9ca45b78f57a5683bf61fa7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9072e185a56d4a89946282f993d31bfc%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_9072e185a56d4a89946282f993d31bfc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaf%C3%A9%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d73028a0b9ca45b78f57a5683bf61fa7.setContent%28html_9072e185a56d4a89946282f993d31bfc%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e5230105d6ed40f08f5425cd89dde7f1.bindPopup%28popup_d73028a0b9ca45b78f57a5683bf61fa7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_76912119cc084673a4ea07e7c3f5d574%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65031871629752%2C-79.37621650642859%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_53f85dee358041d2947e321e314fbdd4%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4bf0d1406176422caa9c2d2828be9ad7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_4bf0d1406176422caa9c2d2828be9ad7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_53f85dee358041d2947e321e314fbdd4.setContent%28html_4bf0d1406176422caa9c2d2828be9ad7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_76912119cc084673a4ea07e7c3f5d574.bindPopup%28popup_53f85dee358041d2947e321e314fbdd4%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e818f4817e08446284e1969d35a33e30%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.66071353922905%2C-79.38549125817697%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_fb27deda633f40c78f083766ff2c8c58%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_725a907062f243ba88af8ed4e737e2df%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_725a907062f243ba88af8ed4e737e2df%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_fb27deda633f40c78f083766ff2c8c58.setContent%28html_725a907062f243ba88af8ed4e737e2df%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_e818f4817e08446284e1969d35a33e30.bindPopup%28popup_fb27deda633f40c78f083766ff2c8c58%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_eba7cac58de5405588c70e4ef566e046%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.644372584148364%2C-79.38306470027423%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_ab5640f3d8c944c8804504412652b880%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_344219cb1f014d57b5cb8a025659397e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_344219cb1f014d57b5cb8a025659397e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_ab5640f3d8c944c8804504412652b880.setContent%28html_344219cb1f014d57b5cb8a025659397e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_eba7cac58de5405588c70e4ef566e046.bindPopup%28popup_ab5640f3d8c944c8804504412652b880%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cb60c9e79a654b3ab70c85203ce3883a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65887244042741%2C-79.39415829049284%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_39d113fee6b84540827009819de0e881%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4650596d93bc43599ed44680e7f34515%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_4650596d93bc43599ed44680e7f34515%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaf%C3%A9%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_39d113fee6b84540827009819de0e881.setContent%28html_4650596d93bc43599ed44680e7f34515%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_cb60c9e79a654b3ab70c85203ce3883a.bindPopup%28popup_39d113fee6b84540827009819de0e881%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fa384441afa641d8b74f74a384b665d0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.64703669923361%2C-79.40087579760974%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_2de790356c724ebaba86f0d05436a269%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6392988839064bf1b34fae4597e3884e%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_6392988839064bf1b34fae4597e3884e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_2de790356c724ebaba86f0d05436a269.setContent%28html_6392988839064bf1b34fae4597e3884e%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fa384441afa641d8b74f74a384b665d0.bindPopup%28popup_2de790356c724ebaba86f0d05436a269%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_851251f49d0c42cba8b5175e844069f3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.645009412720164%2C-79.38581150464259%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d34fb19040cf44dc906e981077215713%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ea8ef9cf9c624e1e9af76f71856b5ee2%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_ea8ef9cf9c624e1e9af76f71856b5ee2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d34fb19040cf44dc906e981077215713.setContent%28html_ea8ef9cf9c624e1e9af76f71856b5ee2%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_851251f49d0c42cba8b5175e844069f3.bindPopup%28popup_d34fb19040cf44dc906e981077215713%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_684de5e6997b4b0a83c705dd90b5fb47%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65505778213131%2C-79.40325401691655%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_f4bdb0d708ea4b278026ef8b6f7b00b7%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2d0abf860f514172885080b00bc38a5c%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_2d0abf860f514172885080b00bc38a5c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_f4bdb0d708ea4b278026ef8b6f7b00b7.setContent%28html_2d0abf860f514172885080b00bc38a5c%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_684de5e6997b4b0a83c705dd90b5fb47.bindPopup%28popup_f4bdb0d708ea4b278026ef8b6f7b00b7%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_de2d1e1ea5624e6fa1a8b9ee0225fe2a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6490525948081%2C-79.37198136360134%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_d1f2d8ca715b41428a2d27b8a808f631%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a325ec7268984e4fbe8a1be687f4a2d3%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_a325ec7268984e4fbe8a1be687f4a2d3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_d1f2d8ca715b41428a2d27b8a808f631.setContent%28html_a325ec7268984e4fbe8a1be687f4a2d3%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_de2d1e1ea5624e6fa1a8b9ee0225fe2a.bindPopup%28popup_d1f2d8ca715b41428a2d27b8a808f631%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fbde9bbe7dc0450998f5b9272624cdfe%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65123%2C-79.368457%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_48c92872dcf4441b9bdef9510506ec05%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b319175d95e24747a5d4d9fc31634e3f%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_b319175d95e24747a5d4d9fc31634e3f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_48c92872dcf4441b9bdef9510506ec05.setContent%28html_b319175d95e24747a5d4d9fc31634e3f%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_fbde9bbe7dc0450998f5b9272624cdfe.bindPopup%28popup_48c92872dcf4441b9bdef9510506ec05%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_03f4d9a9635d400aa7c3f8ad16a07b7f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.657971%2C-79.399795%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%22bubblingMouseEvents%22%3A%20true%2C%0A%20%20%22color%22%3A%20%22blue%22%2C%0A%20%20%22dashArray%22%3A%20null%2C%0A%20%20%22dashOffset%22%3A%20null%2C%0A%20%20%22fill%22%3A%20true%2C%0A%20%20%22fillColor%22%3A%20%22blue%22%2C%0A%20%20%22fillOpacity%22%3A%200.6%2C%0A%20%20%22fillRule%22%3A%20%22evenodd%22%2C%0A%20%20%22lineCap%22%3A%20%22round%22%2C%0A%20%20%22lineJoin%22%3A%20%22round%22%2C%0A%20%20%22opacity%22%3A%201.0%2C%0A%20%20%22radius%22%3A%205%2C%0A%20%20%22stroke%22%3A%20true%2C%0A%20%20%22weight%22%3A%203%0A%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_1f0f4b4ce93e4efbaf0e37354abe5653%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20popup_06e30ed144134b079ad453bd51e32c7c%20%3D%20L.popup%28%7BmaxWidth%3A%20%27300%27%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20var%20html_31598a2a41944b7abb0315adbdae1ab7%20%3D%20%24%28%27%3Cdiv%20id%3D%22html_31598a2a41944b7abb0315adbdae1ab7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECoffee%20Shop%3C/div%3E%27%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20popup_06e30ed144134b079ad453bd51e32c7c.setContent%28html_31598a2a41944b7abb0315adbdae1ab7%29%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20circle_marker_03f4d9a9635d400aa7c3f8ad16a07b7f.bindPopup%28popup_06e30ed144134b079ad453bd51e32c7c%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



**Plotting the most wide-spread Coffee Shops of Toronto**


```python
plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('Number of Coffee Shops in Toronto City')
#On x-axis
plt.xlabel('Street Name', fontsize = 15)
#On y-axis
plt.ylabel('No.of Coffee Houses', fontsize=15)
#giving a bar plot
dataframe_filtered.groupby('name')['name'].count().nlargest(5).plot(kind='bar')
#legend
plt.legend()
#displays the plot
plt.show()
```


![png](output_56_0.png)


# Conclusion:

Timothy's World Coffee has the most number of outlets around the neighbourhood of Toronto, Canada. 

Balzac's Coffee stands at second place. 

In order to compete with the best, a new cafe can be established around these two coffee houses, so that maximum crowd exposure can be established.


# **Limitations:**

The Results are highly dependent on Foursquare API.
