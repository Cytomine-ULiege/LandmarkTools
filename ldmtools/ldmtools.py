# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import numpy as np
import imageio
import scipy.ndimage as snd
from multiprocessing import Pool
import os
import joblib

"""
This function extracts the coordinates of a given term from an offline
cytomine images/coordinates repository.
"""


def getcoords(repository, termid):

	if not repository.endswith('/'):
		repository += '/'

	x  = []
	y  = []
	xp = []
	yp = []
	im = []
	
	for f in os.listdir(repository):
		if f.endswith('.txt'):
			filename = repository+f
			F = open(filename,'r')
			L = F.readlines()
			imageid = int(f.rstrip('.txt'))
			for j in range(len(L)):
				line = L[j].rstrip()
				v = line.split(' ')
				if(int(v[0])==termid):
					x.append(int(float(v[1])))
					y.append(int(float(v[2])))
					xp.append(float(v[3]))
					yp.append(float(v[4]))
					im.append(imageid)
			F.close()
	return np.array(x),np.array(y),np.array(xp),np.array(yp),np.array(im)


def getcoordsim(repository, termid, ims):
	if not repository.endswith('/'):
		repository += '/'

	x = []
	y = []
	xp = []
	yp = []
	im = []
	i = 0
	H = {}
	for i in range(len(ims)):
		H[ims[i]]=i

	x = np.zeros(len(ims))
	y = np.zeros(len(ims))
	xp = np.zeros(len(ims))
	yp = np.zeros(len(ims))

	for f in os.listdir(repository):
		if f.endswith('.txt'):
			filename = repository + f
			F = open(filename, 'r')
			L = F.readlines()
			imageid = int(f.rstrip('.txt'))
			if(imageid in H):
				for j in range(len(L)):
					line = L[j].rstrip()
					v = line.split(' ')
					if (int(v[0]) == termid):
						x[H[imageid]] = int(float(v[1]))
						y[H[imageid]] = int(float(v[2]))
						xp[H[imageid]] = float(v[3])
						yp[H[imageid]] = float(v[4])
			F.close()

	return x, y, xp, yp

def getallcoords(repository):
	if not repository.endswith('/'):
		repository += '/'
	term_to_i = {}
	i_to_term = {}
	nims = len(os.listdir(repository))
	files = os.listdir(repository)
	F = open(repository+files[0])
	lines = F.readlines()
	nldms = len(lines)
	i = 0
	for l in lines:
		v = l.rstrip('\n').split(' ')
		id_term = int(v[0])
		term_to_i[id_term] = i
		i_to_term[i] = id_term
		i += 1

	F.close()

	X = np.zeros((nims,nldms))
	Y = np.zeros((nims,nldms))
	Xr = np.zeros((nims,nldms))
	Yr = np.zeros((nims,nldms))

	ims = []
	im = 0
	for f in os.listdir(repository):
		filename = repository+f
		F = open(filename,'r')
		L = F.readlines()
		for l in L:
			v = l.rstrip('\n').split(' ')
			id_term = int(v[0])

			X[im,term_to_i[id_term]] = float(v[1])
			Y[im,term_to_i[id_term]] = float(v[2])
			Xr[im,term_to_i[id_term]] = float(v[3])
			Yr[im,term_to_i[id_term]] = float(v[4])
		F.close()
		ims.append(int(f.rstrip('.txt')))
		im+=1

	return X,Y,Xr,Yr,ims,term_to_i,i_to_term

"""
This function returns a 8 bit grey-value image given its identifier in the 
offline cytomine repository.
"""

def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray.astype(rgb.dtype)

def readimage(repository,idimage,image_type='jpg'):
	if(not repository.endswith('/')):
		repository = repository+'/'

	if(image_type=='png'):
		IM = rgb2gray(imageio.imread('%s%d.png'%(repository,idimage)))
	elif(image_type=='bmp'):
		IM = rgb2gray(imageio.imread('%s%d.bmp'%(repository,idimage)))
	elif(image_type=='jpg'):
		IM = rgb2gray(imageio.imread('%s%d.jpg'%(repository,idimage)))
	elif(image_type=='tif'):
		IM = rgb2gray(imageio.imread('%s%d.tif'%(repository,idimage)))
	IM = np.double(IM)
	IM = IM-np.mean(IM)
	IM = IM/np.std(IM)
	return IM

"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""
def searchpoint(repository,current,clf,mx,my,cm,depths,window_size,image_type,npred):
	simage = readimage(repository,current,image_type)
	(height,width) = simage.shape
	P = np.random.multivariate_normal([mx,my],cm,npred)	
	x_v = np.round(P[:,0]*width)
	y_v = np.round(P[:,1]*height)
	height=height-1
	width=width-1
	
	n = len(x_v)
	pos = 0
	
	maxprob = -1
	maxx = []
	maxy = []
	
	#maximum number of points considered at once in order to not overload the
	#memory.
	step = 100000

	for index in range(len(x_v)):
		xv = x_v[index]
		yv = y_v[index]
		if(xv<0):
			x_v[index] = 0
		if(yv<0):
			y_v[index] = 0
		if(xv>width):
			x_v[index] = width
		if(yv>height):
			y_v[index] = height
	
	while(pos<n):
		xp = np.array(x_v[pos:min(n,pos+step)])
		yp = np.array(y_v[pos:min(n,pos+step)])
		
		DATASET = build_dataset_image(simage,window_size,xp,yp,depths)
		pred = clf.predict_proba(DATASET)
		pred = pred[:,1]
		maxpred = np.max(pred)
		if(maxpred>=maxprob):
			positions = np.where(pred==maxpred)
			positions = positions[0]
			xsup = xp[positions]
			ysup = yp[positions]
			if(maxpred>maxprob):
				maxprob = maxpred
				maxx = xsup
				maxy = ysup
			else:
				maxx = np.concatenate((maxx,xsup))
				maxy = np.concatenate((maxy,ysup))
		pos = pos+step
				
	return np.median(maxx),np.median(maxy),height-np.median(maxy)

"""
0-padding of an image IM of wp pixels on all sides
"""
def makesize(IM,wp):
	(h,w) = IM.shape
	IM2 = np.zeros((h+2*wp,w+2*wp))
	IM2[wp:wp+h,wp:wp+w] = IM
	return IM2


def build_integral_slice(img):
	img = np.double(img)
	img = img - np.mean(img)
	img = img / np.std(img)
	intg = np.zeros(img.shape)
	(h, w) = img.shape
	intg[0, 0] = img[0, 0]

	for i in range(1, w):
		intg[0, i] = intg[0, i - 1] + img[0, i]
	for i in range(1, h):
		intg[i, 0] = intg[i - 1, 0] + img[i, 0]

	for i in range(1, h):
		i1 = i - 1
		for j in range(1, w):
			j1 = j - 1
			intg[i, j] = img[i, j] + intg[i1, j] + intg[i, j1] - intg[i1, j1]

	return intg


def generate_2_horizontal(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w, n)
	coords[:, 1] = np.random.randint(-w, w + 1, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(1, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(0, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_2_vertical(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w + 1, n)
	coords[:, 1] = np.random.randint(-w, w, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(0, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(1, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_3_horizontal(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w - 1, n)
	coords[:, 1] = np.random.randint(-w, w + 1, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(2, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(0, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_3_vertical(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w + 1, n)
	coords[:, 1] = np.random.randint(-w, w - 1, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(0, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(2, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_square(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w, n)
	coords[:, 1] = np.random.randint(-w, w, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(1, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(1, w + 1 - coords[i, 1]) for i in range(n)]
	return coords

def generate_2d_coordinates_horizontal(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 6))
	y = np.zeros((n, 6))

	w = np.floor(0.5 * ((coords[:, 2] - coords[:, 0]) + 1.)).astype('int')
	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = x[:, 0] + w
	y[:, 1] = y[:, 0]
	x[:, 2] = coords[:, 2]
	y[:, 2] = y[:, 1]
	x[:, 3] = x[:, 0]
	y[:, 3] = coords[:, 3]
	x[:, 4] = x[:, 1]
	y[:, 4] = y[:, 3]
	x[:, 5] = x[:, 2]
	y[:, 5] = y[:, 4]

	return x.astype('int'), y.astype('int')


def generate_2d_coordinates_vertical(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 6))
	y = np.zeros((n, 6))

	w = np.floor(0.5 * ((coords[:, 3] - coords[:, 1]) + 1)).astype('int')
	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = coords[:, 2]
	y[:, 1] = y[:, 0]
	x[:, 2] = x[:, 0]
	y[:, 2] = y[:, 0] + w
	x[:, 3] = x[:, 1]
	y[:, 3] = y[:, 2]
	x[:, 4] = x[:, 2]
	y[:, 4] = coords[:, 3]
	x[:, 5] = x[:, 3]
	y[:, 5] = y[:, 4]

	return x.astype('int'), y.astype('int')


def generate_3d_coordinates_horizontal(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 8))
	y = np.zeros((n, 8))
	w = np.floor(((coords[:, 2] - coords[:, 0]) + 1.) / 3.).astype('int')

	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = x[:, 0] + w
	y[:, 1] = y[:, 0]
	x[:, 2] = x[:, 1] + w
	y[:, 2] = y[:, 0]
	x[:, 3] = coords[:, 2]
	y[:, 3] = y[:, 0]
	x[:, 4] = x[:, 0]
	y[:, 4] = coords[:, 3]
	x[:, 5] = x[:, 1]
	y[:, 5] = y[:, 4]
	x[:, 6] = x[:, 2]
	y[:, 6] = y[:, 4]
	x[:, 7] = x[:, 3]
	y[:, 7] = y[:, 4]

	return x.astype('int'), y.astype('int')


def generate_3d_coordinates_vertical(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 8))
	y = np.zeros((n, 8))
	w = np.floor(((coords[:, 3] - coords[:, 1]) + 1.) / 3.).astype('int')

	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = coords[:, 2]
	y[:, 1] = y[:, 0]
	x[:, 2] = x[:, 0]
	y[:, 2] = y[:, 0] + w
	x[:, 3] = x[:, 1]
	y[:, 3] = y[:, 2]
	x[:, 4] = x[:, 2]
	y[:, 4] = y[:, 2] + w
	x[:, 5] = x[:, 3]
	y[:, 5] = y[:, 4]
	x[:, 6] = x[:, 4]
	y[:, 6] = coords[:, 3]
	x[:, 7] = x[:, 5]
	y[:, 7] = y[:, 6]

	return x.astype('int'), y.astype('int')


def generate_square_coordinates(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 9))
	y = np.zeros((n, 9))

	wx = np.floor(0.5 * ((coords[:, 2] - coords[:, 0]) + 1.)).astype('int')
	wy = np.floor(0.5 * ((coords[:, 3] - coords[:, 1]) + 1.)).astype('int')

	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1

	x[:, 1] = x[:, 0] + wx
	y[:, 1] = y[:, 0]

	x[:, 2] = coords[:, 2]
	y[:, 2] = y[:, 0]

	x[:, 3] = x[:, 0]
	y[:, 3] = y[:, 0] + wy

	x[:, 4] = x[:, 1]
	y[:, 4] = y[:, 3]

	x[:, 5] = x[:, 2]
	y[:, 5] = y[:, 4]

	x[:, 6] = x[:, 3]
	y[:, 6] = coords[:, 3]

	x[:, 7] = x[:, 4]
	y[:, 7] = y[:, 6]

	x[:, 8] = x[:, 5]
	y[:, 8] = y[:, 6]

	return x.astype('int'), y.astype('int')

"""
Build the dataset on a single image
"""
def build_dataset_image( IM, wp, x_v, y_v, feature_type, feature_parameters, depths):
	feature_type = feature_type.lower()
	if(feature_type=='raw'):
		swp = (2*wp)**2
		ndata=len(x_v)
		dwp=2*wp
		ndepths = len(depths)
	
		DATASET = np.zeros((ndata,swp*ndepths))
		images = {}
		for z in range(ndepths):
			images[z] = makesize(snd.zoom(IM,depths[z]),wp)
		
		X = [[int(x*depths[z]) for x in x_v] for z in range(ndepths)]
		Y = [[int(y*depths[z]) for y in y_v] for z in range(ndepths)]

		cub = np.zeros((ndepths,dwp,dwp))
	
		for j in range(ndata):
			x = x_v[j]		
			y = y_v[j]
			for z in range(ndepths):
				im = images[z]
				cub[z,:,:] = im[Y[z][j]:Y[z][j]+dwp,X[z][j]:X[z][j]+dwp]
				cub[z,:,:] = im[Y[z][j]:Y[z][j]+dwp,X[z][j]:X[z][j]+dwp] - IM[y, x]
			DATASET[j,:] = cub.flatten()
		return DATASET

	elif(feature_type=='sub'):
		swp = (2*wp)**2
		ndata=len(x_v)
		dwp=2*wp
		ndepths = len(depths)
	
		DATASET = np.zeros((ndata,swp*ndepths))

		images = {}
		for z in range(ndepths):
			images[z] = makesize(snd.zoom(IM,depths[z]),wp)
		
		X = [[int(x*depths[z]) for x in x_v] for z in range(ndepths)]
		Y = [[int(y*depths[z]) for y in y_v] for z in range(ndepths)]
		cub = np.zeros((ndepths,dwp,dwp))
	
		for j in range(ndata):
			x = int(x_v[j])
			y = int(y_v[j])
			for z in range(ndepths):
				im = images[z]
				cub[z,:,:] = im[Y[z][j]:Y[z][j]+dwp,X[z][j]:X[z][j]+dwp]-IM[y,x]
			DATASET[j,:] = cub.flatten()
		return DATASET
	elif(feature_type=='haar'):
		(coords_h2,coords_v2,coords_h3,coords_v3,coords_sq) = feature_parameters
		xo = np.array(x_v)
		yo = np.array(y_v)
		n_h2 = coords_h2.shape[0]
		n_v2 = coords_v2.shape[0]
		n_h3 = coords_h3.shape[0]
		n_v3 = coords_v3.shape[0]
		n_sq = coords_sq.shape[0]

		ndata = xo.size
		coords = np.zeros((ndata,4))
		dataset = np.zeros((ndata,(n_h2+n_v2+n_h3+n_v3+n_sq)*depths.size))

		feature_index = 0
	
		for resolution in depths:

			if(resolution==1):
				rimg = IM
			else:
				rimg = snd.zoom(IM,resolution)
		
			intg = build_integral_slice(rimg)
		
			pad_intg = makesize(intg,1)

			x = np.round((xo*resolution)+1).astype(int)
			y = np.round((yo*resolution)+1).astype(int)
			(h,w) = pad_intg.shape
			h-=1
			w-=1

			for i in range(n_h2):
				coords[:,0] = (x+coords_h2[i,0])
				coords[:,1] = (y+coords_h2[i,1])
				coords[:,2] = (x+coords_h2[i,2])
				coords[:,3] = (y+coords_h2[i,3])
				(xc,yc) = generate_2d_coordinates_horizontal(coords)
				xc = xc.clip(min=0,max=w)
				yc = yc.clip(min=0,max=h)
				zero   = pad_intg[yc[:,0],xc[:,0]]
				un     = pad_intg[yc[:,1],xc[:,1]]
				deux   = pad_intg[yc[:,2],xc[:,2]]
				trois  = pad_intg[yc[:,3],xc[:,3]]
				quatre = pad_intg[yc[:,4],xc[:,4]]
				cinq   = pad_intg[yc[:,5],xc[:,5]]
				dataset[:,feature_index] = zero+(2*un)+(-deux)+trois+(-2*quatre)+cinq
				feature_index += 1
	
			for i in range(n_v2):
				coords[:,0] = x+coords_v2[i,0]
				coords[:,1] = y+coords_v2[i,1]
				coords[:,2] = x+coords_v2[i,2]
				coords[:,3] = y+coords_v2[i,3]
				(xc,yc) = generate_2d_coordinates_vertical(coords)
				xc = xc.clip(min=0,max=w)
				yc = yc.clip(min=0,max=h)
				zero   = pad_intg[yc[:,0],xc[:,0]]
				un     = pad_intg[yc[:,1],xc[:,1]]
				deux   = pad_intg[yc[:,2],xc[:,2]]
				trois  = pad_intg[yc[:,3],xc[:,3]]
				quatre = pad_intg[yc[:,4],xc[:,4]]
				cinq   = pad_intg[yc[:,5],xc[:,5]]
				dataset[:,feature_index] = zero+(-un)+(-2*deux)+(2*trois)+quatre-cinq
				feature_index+=1
	
			for i in range(n_h3):
				coords[:,0] = x+coords_h3[i,0]
				coords[:,1] = y+coords_h3[i,1]
				coords[:,2] = x+coords_h3[i,2]
				coords[:,3] = y+coords_h3[i,3]
				(xc,yc) = generate_3d_coordinates_horizontal(coords)
				xc = xc.clip(min=0,max=w)
				yc = yc.clip(min=0,max=h)
				zero   = pad_intg[yc[:,0],xc[:,0]]
				un     = pad_intg[yc[:,1],xc[:,1]]
				deux   = pad_intg[yc[:,2],xc[:,2]]
				trois  = pad_intg[yc[:,3],xc[:,3]]
				quatre = pad_intg[yc[:,4],xc[:,4]]
				cinq   = pad_intg[yc[:,5],xc[:,5]]
				six    = pad_intg[yc[:,6],xc[:,6]]
				sept   = pad_intg[yc[:,7],xc[:,7]]
				dataset[:,feature_index] = zero+(-2*un)+(2*deux)+(-trois)+(-quatre)+(2*cinq)+(-2*six)+sept
				feature_index += 1
		
			for i in range(n_v3):
				coords[:,0] = x+coords_v3[i,0]
				coords[:,1] = y+coords_v3[i,1]
				coords[:,2] = x+coords_v3[i,2]
				coords[:,3] = y+coords_v3[i,3]
				(xc,yc) = generate_3d_coordinates_vertical(coords)
				xc = xc.clip(min=0,max=w)
				yc = yc.clip(min=0,max=h)
				zero   = pad_intg[yc[:,0],xc[:,0]]
				un     = pad_intg[yc[:,1],xc[:,1]]
				deux   = pad_intg[yc[:,2],xc[:,2]]
				trois  = pad_intg[yc[:,3],xc[:,3]]
				quatre = pad_intg[yc[:,4],xc[:,4]]
				cinq   = pad_intg[yc[:,5],xc[:,5]]
				six    = pad_intg[yc[:,6],xc[:,6]]
				sept   = pad_intg[yc[:,7],xc[:,7]]
				dataset[:,feature_index] = zero+(-un)+(-2*deux)+(2*trois)+(2*quatre)+(-2*cinq)+(-six)+sept
				feature_index += 1
		
			for i in range(n_sq):
				coords[:,0] = x+coords_sq[i,0]
				coords[:,1] = y+coords_sq[i,1]
				coords[:,2] = x+coords_sq[i,2]
				coords[:,3] = y+coords_sq[i,3]
				(xc,yc) = generate_square_coordinates(coords)
				xc = xc.clip(min=0,max=w)
				yc = yc.clip(min=0,max=h)
				zero   = pad_intg[yc[:,0],xc[:,0]]
				un     = pad_intg[yc[:,1],xc[:,1]]
				deux   = pad_intg[yc[:,2],xc[:,2]]
				trois  = pad_intg[yc[:,3],xc[:,3]]
				quatre = pad_intg[yc[:,4],xc[:,4]]
				cinq   = pad_intg[yc[:,5],xc[:,5]]
				six    = pad_intg[yc[:,6],xc[:,6]]
				sept   = pad_intg[yc[:,7],xc[:,7]]
				huit   = pad_intg[yc[:,8],xc[:,8]]
				dataset[:,feature_index] = zero+(-2*un)+deux+(-2*trois)+(4*quatre)+(-2*cinq)+six+(-2*sept)+huit
				feature_index += 1
		return dataset
	elif(feature_type=='gaussian'):
		xo = np.array(x_v)
		yo = np.array(y_v)
		feature_offsets = feature_parameters
		dataset = np.zeros((xo.size, feature_offsets[:, 0].size * depths.size))
		j = 0
		for kl in range(depths.size):
			resolution = depths[kl]
			rimg = snd.zoom(IM, resolution)
			rimg = makesize(rimg, 1)
			x = np.round((xo * resolution) + 1).astype(int)
			y = np.round((yo * resolution) + 1).astype(int)
			(h, w) = rimg.shape
			original_values = rimg[y, x]
			for i in range(feature_offsets[:, 0].size):
				dataset[:, j] = original_values - rimg[
					(y + feature_offsets[i, 1]).clip(min=0, max=h - 1), (x + feature_offsets[i, 0]).clip(min=0,
					                                                                                     max=w - 1)]
				j = j + 1
		return dataset
	return None

def rotate_coordinates(repository,num,x,y,angle,image_type):
	image = readimage(repository,num,image_type)
	if(angle!=0):
		image_rotee = snd.rotate(image,-angle)
		(h,w) = image.shape
		(hr,wr) = image_rotee.shape
		angle_rad = angle*(np.pi/180.)
		c = np.cos(angle_rad)
		s = np.sin(angle_rad)
		x = x-(w/2.)
		y = y-(h/2.)
		xr = ((x*c)-(y*s))+(wr/2.)
		yr = ((x*s)+(y*c))+(hr/2.)
		return xr.tolist(),yr.tolist(),image_rotee
	else:
		return x.tolist(),y.tolist(),image
	
def dataset_image_rot(repository,Xc,Yc,R,RMAX,proportion,step,i,ang,feature_type,feature_parameters,depths,window_size,image_type):
		print("IMAGE %d"%i)
		x_v = []
		y_v = []
		REP = []
		IMGS = []
		deuxpi = 2.*np.pi
		for x in np.arange(np.int(Xc)-R,np.int(Xc)+R+1,step):
			for y in np.arange(np.int(Yc)-R,np.int(Yc)+R+1,step):
				if(np.linalg.norm([Xc-x,Yc-y])<=R):
					x_v.append(x)
					y_v.append(y)
					REP.append(1)
					IMGS.append(i)
		
		n = len(x_v)
		image = readimage(repository,i,image_type)
		(height,width)=image.shape
		height=height-1
		width=width-1
		for t in range(int(round(proportion*n))):
			angle = np.random.ranf()*deuxpi
			r = R + (np.random.ranf()*(RMAX-R))
			tx = int(r*np.cos(angle))
			ty = int(r*np.sin(angle))
			x_v.append(min(width,Xc+tx))
			y_v.append(min(height,Yc+ty))
			REP.append(0)
			IMGS.append(i)
			
		(x_r,y_r,im) = rotate_coordinates(repository,i,np.round(np.array(x_v)),np.round(np.array(y_v)),ang,image_type)
		(hr,wr) = im.shape
		hr -= 1
		wr -= 1
		
		x_r = np.round(x_r)
		y_r = np.round(y_r)
		
		for index in range(len(x_r)):
			xr = x_r[index]
			yr = y_r[index]
			if(xr<0):
				x_r[index] = 0
			if(yr<0):
				y_r[index] = 0
			if(xr>wr):
				x_r[index] = wr
			if(yr>hr):
				y_r[index] = hr
				
		return build_dataset_image(im,window_size,x_r,y_r,feature_type,feature_parameters,depths),REP,IMGS

def mp_helper_rot(job_args):
	return dataset_image_rot(*job_args)
	
def build_datasets_rot_mp(repository,imc,Xc,Yc,R,RMAX,proportion,step,ang,window_size,feature_type,feature_parameters,depths,nimages,image_type, njobs):

	p = Pool(njobs) 
	job_args = [(repository, Xc[i], Yc[i], R, RMAX, proportion, step, imc[i], (np.random.ranf()*2*ang)-ang, feature_type, feature_parameters, depths, window_size, image_type) for i in range(nimages)]
	T = p.map(mp_helper_rot,job_args)
	p.close()
	p.join()
	return T

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

def pad_integral(intg):
	(h,w) = intg.shape
	nintg = np.zeros((h+1,w+1))
	nintg[1:,1:]=intg
	return nintg

def compute_features(intg, x, y, coords_h2, coords_v2, coords_h3, coords_v3, coords_sq):
	pad_intg = pad_integral(intg)
	x = x + 1
	y = y + 1
	(h, w) = pad_intg.shape
	h -= 1
	w -= 1

	(n_h2, quatre) = coords_h2.shape
	(n_v2, quatre) = coords_v2.shape
	(n_h3, quatre) = coords_h3.shape
	(n_v3, quatre) = coords_v3.shape
	(n_sq, quatre) = coords_sq.shape

	ndata = x.size
	coords = np.zeros((ndata, 4))
	dataset = np.zeros((ndata, n_h2 + n_v2 + n_h3 + n_v3 + n_sq))
	feature_index = 0

	for i in range(n_h2):
		coords[:, 0] = (x + coords_h2[i, 0])
		coords[:, 1] = (y + coords_h2[i, 1])
		coords[:, 2] = (x + coords_h2[i, 2])
		coords[:, 3] = (y + coords_h2[i, 3])
		(xc, yc) = generate_2d_coordinates_horizontal(coords)
		xc = xc.clip(min=0, max=w)
		yc = yc.clip(min=0, max=h)
		zero = pad_intg[yc[:, 0], xc[:, 0]]
		un = pad_intg[yc[:, 1], xc[:, 1]]
		deux = pad_intg[yc[:, 2], xc[:, 2]]
		trois = pad_intg[yc[:, 3], xc[:, 3]]
		quatre = pad_intg[yc[:, 4], xc[:, 4]]
		cinq = pad_intg[yc[:, 5], xc[:, 5]]
		dataset[:, feature_index] = zero + (2 * un) + (-deux) + trois + (-2 * quatre) + cinq
		feature_index += 1

	for i in range(n_v2):
		coords[:, 0] = x + coords_v2[i, 0]
		coords[:, 1] = y + coords_v2[i, 1]
		coords[:, 2] = x + coords_v2[i, 2]
		coords[:, 3] = y + coords_v2[i, 3]
		(xc, yc) = generate_2d_coordinates_vertical(coords)
		xc = xc.clip(min=0, max=w)
		yc = yc.clip(min=0, max=h)
		zero = pad_intg[yc[:, 0], xc[:, 0]]
		un = pad_intg[yc[:, 1], xc[:, 1]]
		deux = pad_intg[yc[:, 2], xc[:, 2]]
		trois = pad_intg[yc[:, 3], xc[:, 3]]
		quatre = pad_intg[yc[:, 4], xc[:, 4]]
		cinq = pad_intg[yc[:, 5], xc[:, 5]]
		dataset[:, feature_index] = zero + (-un) + (-2 * deux) + (2 * trois) + quatre - cinq
		feature_index += 1

	for i in range(n_h3):
		coords[:, 0] = x + coords_h3[i, 0]
		coords[:, 1] = y + coords_h3[i, 1]
		coords[:, 2] = x + coords_h3[i, 2]
		coords[:, 3] = y + coords_h3[i, 3]
		(xc, yc) = generate_3d_coordinates_horizontal(coords)
		xc = xc.clip(min=0, max=w)
		yc = yc.clip(min=0, max=h)
		zero = pad_intg[yc[:, 0], xc[:, 0]]
		un = pad_intg[yc[:, 1], xc[:, 1]]
		deux = pad_intg[yc[:, 2], xc[:, 2]]
		trois = pad_intg[yc[:, 3], xc[:, 3]]
		quatre = pad_intg[yc[:, 4], xc[:, 4]]
		cinq = pad_intg[yc[:, 5], xc[:, 5]]
		six = pad_intg[yc[:, 6], xc[:, 6]]
		sept = pad_intg[yc[:, 7], xc[:, 7]]
		dataset[:, feature_index] = zero + (-2 * un) + (2 * deux) + (-trois) + (-quatre) + (2 * cinq) + (
					-2 * six) + sept
		feature_index += 1

	for i in range(n_v3):
		coords[:, 0] = x + coords_v3[i, 0]
		coords[:, 1] = y + coords_v3[i, 1]
		coords[:, 2] = x + coords_v3[i, 2]
		coords[:, 3] = y + coords_v3[i, 3]
		(xc, yc) = generate_3d_coordinates_vertical(coords)
		xc = xc.clip(min=0, max=w)
		yc = yc.clip(min=0, max=h)
		zero = pad_intg[yc[:, 0], xc[:, 0]]
		un = pad_intg[yc[:, 1], xc[:, 1]]
		deux = pad_intg[yc[:, 2], xc[:, 2]]
		trois = pad_intg[yc[:, 3], xc[:, 3]]
		quatre = pad_intg[yc[:, 4], xc[:, 4]]
		cinq = pad_intg[yc[:, 5], xc[:, 5]]
		six = pad_intg[yc[:, 6], xc[:, 6]]
		sept = pad_intg[yc[:, 7], xc[:, 7]]
		dataset[:, feature_index] = zero + (-un) + (-2 * deux) + (2 * trois) + (2 * quatre) + (-2 * cinq) + (
			-six) + sept
		feature_index += 1

	for i in range(n_sq):
		coords[:, 0] = x + coords_sq[i, 0]
		coords[:, 1] = y + coords_sq[i, 1]
		coords[:, 2] = x + coords_sq[i, 2]
		coords[:, 3] = y + coords_sq[i, 3]
		(xc, yc) = generate_square_coordinates(coords)
		xc = xc.clip(min=0, max=w)
		yc = yc.clip(min=0, max=h)
		zero = pad_intg[yc[:, 0], xc[:, 0]]
		un = pad_intg[yc[:, 1], xc[:, 1]]
		deux = pad_intg[yc[:, 2], xc[:, 2]]
		trois = pad_intg[yc[:, 3], xc[:, 3]]
		quatre = pad_intg[yc[:, 4], xc[:, 4]]
		cinq = pad_intg[yc[:, 5], xc[:, 5]]
		six = pad_intg[yc[:, 6], xc[:, 6]]
		sept = pad_intg[yc[:, 7], xc[:, 7]]
		huit = pad_intg[yc[:, 8], xc[:, 8]]
		dataset[:, feature_index] = zero + (-2 * un) + deux + (-2 * trois) + (4 * quatre) + (-2 * cinq) + six + (
					-2 * sept) + huit
		feature_index += 1

	return dataset
