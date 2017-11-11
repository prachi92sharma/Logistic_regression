#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division  # Python 2 users only
import re
from nltk import word_tokenize
import io
import numpy
import os,sys,re,string
import collections
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
import h5py

stopwords1 = set(stopwords.words('english'))
tardict = defaultdict(int)
word_dict = defaultdict(int)
#count = defaultdict(lambda: defaultdict(int))  #for making defaultdict callable
#fpath = os.path.join("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt")
fpath = os.path.join("/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt")
with open(fpath, "r") as script:
	filelines =script.readlines()[3:]
	#filelines=[filelines[0]]
	for line in filelines:
		    values = line.split("\t")
		    labels=values[0].rstrip().split(",")
		    words = re.sub("\d+", "", values[1].rsplit('"',1)[0].split('"',1)[1])
		    regex = r'(\w*)'
		    wlist=re.findall(regex,words)
		    while '' in wlist:
		       wlist.remove('')
		    wlist=map(str.lower,wlist)
		    wlist = [w for w in wlist if not w in stopwords1]
		    for target in labels:
			tardict[target] += 1
		    for word in wlist:
			if (len(word)) > 3:
				word_dict[word] += 1  

#print(len(word_dict))

#giving id's to each word in vocabulory and label dictionary


k=0
for target in tardict.keys():
	tardict[target]=k
	k=k+1

k=0
for word in word_dict.keys():
	if word_dict[word]>105:
		word_dict[word]=k
		k=k+1
	else:
		del word_dict[word]
	
#print(len(word_dict))

#giving zeros and ones to label and words in each document

print(len(word_dict))
#training_word = 
#fpath = os.path.join("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt")
fpath = os.path.join("/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt")
with open(fpath, "r") as script:
	train_filelines =script.readlines()[3:]

tr_w = np.zeros((len(train_filelines),len(word_dict)),dtype = np.float32)
tr_l = np.zeros((len(train_filelines),len(tardict)), dtype = np.float32)

i = 0
for line in train_filelines:
	values = line.split("\t")
	labels=values[0].rstrip().split(",")
	words = re.sub("\d+", "", values[1].rsplit('"',1)[0].split('"',1)[1])
        regex = r'(\w*)'
	wlist=re.findall(regex,words)
	while '' in wlist:
		wlist.remove('')
	wlist=map(str.lower,wlist)
        wlist = [w for w in wlist if not w in stopwords1]
	for w in wlist:
		if w in word_dict:
			tr_w[i,word_dict[w]]=1
	for l in labels:
		tr_l[i,tardict[l]] = 1
	tr_l[i,:] = tr_l[i,:]/np.sum(tr_l[i,:])
		
	i=i+1

print(len(word_dict))      

fpath = os.path.join("/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt")
with open(fpath, "r") as script:
	test_filelines =script.readlines()[3:]
test_w = np.zeros((len(test_filelines),len(word_dict)),dtype = np.float32)
test_l = np.zeros((len(test_filelines),len(tardict)),dtype = np.float32)

i=0
for line in test_filelines:
	values = line.split("\t")
	labels=values[0].rstrip().split(",")
	words = re.sub("\d+", "", values[1].rsplit('"',1)[0].split('"',1)[1])
        regex = r'(\w*)'
	wlist=re.findall(regex,words)
	while '' in wlist:
		wlist.remove('')
	wlist=map(str.lower,wlist)
        wlist = [w for w in wlist if not w in stopwords1]
	
	for w in wlist:
		test_w[i,word_dict[w]]=1
	for l in labels:
		test_l[i,tardict[l]] = 1
	test_l[i,:] = test_l[i,:]/np.sum(test_l[i,:])
		
	i=i+1


#print(np.sum(test_l[4,:]))



h5f1 = h5py.File('train_label.h5', 'w')
h5f1.create_dataset('d1', data=tr_l)
h5f2 = h5py.File('test_label.h5', 'w')
h5f2.create_dataset('d2', data=test_l)
h5f3 = h5py.File('train_word.h5', 'w')
h5f3.create_dataset('d3', data=tr_w)
h5f4 = h5py.File('test_word.h5', 'w')
h5f4.create_dataset('d4', data=test_w)
h5f1.close()
h5f2.close()
h5f3.close()
h5f4.close()


print(tr_w.shape)
print(tr_l.shape)
print(test_w.shape)
print(test_l.shape)











