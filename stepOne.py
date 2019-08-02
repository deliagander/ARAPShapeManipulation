import numpy as np
import cv2

def findNeighbour(edge, faces):
	# find the four neighbouring vertices of the edge
	neighbours = [np.nan,np.nan]
	count = 0
	for i,face in enumerate(faces):
		if np.any(face == edge[0]):
			if np.any(face == edge[1]):
				neighbourI = np.where(face[np.where(face!=edge[0])]!=edge[1])
				neighbourI = neighbourI[0][0]
				n = face[np.where(face!=edge[0])]
				neighbours[count] = int(n[neighbourI])
				count += 1
	lI,rI = neighbours
	return lI, rI	


def computeG(vertices, edges, faces):
	# G = np.zeros((np.size(edges,0), 8,4))
	gCalc = np.zeros((np.size(edges,0), 2,8))
	GI = np.zeros((np.size(edges,0),4))
	for k,edge in enumerate(edges):
		iI = int(edge[0])
		jI = int(edge[1])
		i = vertices[iI-1,:]
		j = vertices[jI-1,:]
		lI, rI = findNeighbour(edge,faces)
		l = vertices[lI-1,:]
		if not np.isnan(rI):
			r = vertices[rI-1,:]

		if np.isnan(rI):
			g = np.array([[i[0], i[1], 1,0], 
				[i[1], -i[0],0,1],
				[j[0], j[1],1,0], 
				[j[1], -j[0],0,1],
				[l[0], l[1],1,0],
				[l[1], -l[0],0,1]])
			# G[k,:,:] = g
			GI[k,:] = [iI, jI, lI, np.nan]
			gTemp = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
			gCalc[k,:,0:6] = gTemp[0:2,:]
        
		else:
			g = np.array([[i[0], i[1], 1,0], 
				[i[1], -i[0],0,1],
				[j[0], j[1], 1,0], 
				[j[1], -j[0],0,1],
				[l[0], l[1], 1,0],
				[l[1], -l[0],0,1],
				[r[0], r[1], 1,0],
				[r[1], -r[0],0,1]])
			# G[k,:,:]
			GI[k,:] = [iI, jI, lI, rI]
			gTemp = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
			gCalc[k,:,:] = gTemp[0:2,:]
	return GI,gCalc

def computeH(edges, gCalc, GI,vertices):
	H = np.zeros((np.size(edges,0)*2,8));
	for k, edge in enumerate(edges):

		ek = vertices[int(edge[1])-1,:] - vertices[int(edge[0])-1,:]
		EK = [[ek[0],ek[1]], [ek[1],-ek[0]]]

		if np.isnan(GI[k,3]):
			oz = [[-1,0,1,0,0,0],
				[0,-1,0,1,0,0]]
			g = gCalc[k,:,0:6]
			# gCalc = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]			
			hCalc = oz - (EK@g)
			H[k*2,0:6] = hCalc[0,:]
			H[k*2+1,0:6] = hCalc[1,:]
		else:
			oz = [[-1,0,1,0,0,0,0,0],
				[0,-1,0,1,0,0,0,0]]
			g = gCalc[k,:,:]
			# gCalc = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
			hCalc = oz - (EK@g)
			H[k*2,:] = hCalc[0,:]
			H[k*2+1,:] = hCalc[1,:]	
	return H


def computeVPrime(edges, vertices, GI, H, C, locations):
	A = np.zeros((np.size(edges, 0)*2 + np.size(C)*2, np.size(vertices, 0)*2))
	b = np.zeros((np.size(edges, 0)*2 + np.size(C)*2,1))

	w = 1000

	vPrime = np.zeros((np.size(vertices,0),2))

	for gIndex, g in enumerate(GI):
		for vIndex, v in enumerate(g):
			if not np.isnan(v):
				v = int(v)-1
				A[gIndex*2,v*2] = H[gIndex*2,vIndex*2]
				A[gIndex*2+1,v*2] = H[gIndex*2+1,vIndex*2]
				A[gIndex*2,v*2+1] = H[gIndex*2,vIndex*2+1]
				A[gIndex*2+1,v*2+1] = H[gIndex*2+1,vIndex*2+1]

	for cIndex, c in enumerate(C):
		A[np.size(edges, 0)*2+cIndex*2,c*2] = w
		A[np.size(edges, 0)*2+cIndex*2+1,c*2+1] = w
		b[np.size(edges, 0)*2 + cIndex*2] = w*locations[cIndex,0]
		b[np.size(edges, 0)*2 + cIndex*2+1] = w*locations[cIndex,1]

	V = np.linalg.lstsq(A.T@A,A.T@b, rcond=None)[0]

	vPrime[:,0] = V[0::2,0]
	vPrime[:,1] = V[1::2,0]

	return vPrime, A, b

def computeVPrimeFast(edges, vertices, C, locations, A,b):
	w = 1000

	vPrime = np.zeros((np.size(vertices,0),2))

	V = np.linalg.lstsq(A.T@A,A.T@b, rcond=None)[0]

	vPrime[:,0] = V[0::2,0]
	vPrime[:,1] = V[1::2,0]

	return vPrime
