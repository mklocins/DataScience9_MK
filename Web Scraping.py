# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:40:17 2015

@author: PoltergistAA
"""

'''OPTIONAL WEB SCRAPING HOMEWORK
First, define a function that accepts an IMDb ID and returns a dictionary of
movie information: title, star_rating, description, content_rating, duration.
The function should gather this information by scraping the IMDb website, not
by calling the OMDb API. (This is really just a wrapper of the web scraping
code we wrote above.)
For example, get_movie_info('tt0111161') should return:
{'content_rating': 'R',
 'description': u'Two imprisoned men bond over a number of years...',
 'duration': 142,
 'star_rating': 9.3,
 'title': u'The Shawshank Redemption'}
Then, open the file imdb_ids.txt using Python, and write a for loop that builds
a list in which each element is a dictionary of movie information.
Finally, convert that list into a DataFrame.'''

import os
import pandas as pd
from bs4 import BeautifulSoup

os.chdir('C:\Users\PoltergistAA\Documents\GitHub\DAT-DC-9\data')

imdb_file = pd.read_table('imdb_ids.txt', header=None, sep='\t')
imdb_file = pd.DataFrame(imdb_file)


def imdb(imdb_id):
    r = requests.get('http://www.imdb.com/title/' + str(imdb_id) + '/')
    b = BeautifulSoup(r.text)
    imdb_dict = {}
    imdb_dict['Content_Rating'] = b.find(name='span', attrs={'itemprop':'contentRating'})
    imdb_dict['Description'] = b.find(name='meta', attrs={'name':'description'})['content']
    imdb_dict['Duration'] = b.find(name='time', attrs={'itemprop':'duration'}).text.replace('min', ' ').strip()
    imdb_dict['Star_Rating'] = b.find(name='div', attrs={'class':'titlePageSprite star-box-giga-star'}).text
    imdb_dict['Title'] = b.find(name='title').text 
    return imdb_dict
    
list_dicts = []

for movie_id in imdb_file[0]:
    list_dicts.append(imdb(str(movie_id)))

movieDF = pd.DataFrame(list_dicts)


