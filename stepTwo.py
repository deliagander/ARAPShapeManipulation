import numpy as np
import cv2

def computeT(edges, gCalc, GI, vPrime):
	T = np.zeros(((np.size(edges, 0)),2,2,))
	for k, edge in enumerate(edges):
		if np.isnan(GI[k,3]):
			v = np.array([[vPrime[int(GI[k,0])-1,0]], 
				[vPrime[int(GI[k,0])-1,1]],
				[vPrime[int(GI[k,1])-1,0]], 
				[vPrime[int(GI[k,1])-1,1]],
				[vPrime[int(GI[k,2])-1,0]], 
				[vPrime[int(GI[k,2])-1,1]]])
			g = gCalc[k,:,0:6]
			# gCalc = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
			t = g@v
			t_temp = (1/np.sqrt(t[0]**2 + t[1]**2)) * [[t[0], t[1]], [-t[1],t[0]]]
			T[k,:,:] = t_temp[:,:,0]
		else:
			v = np.array([[vPrime[int(GI[k,0])-1,0]], 
				[vPrime[int(GI[k,0])-1,1]],
				[vPrime[int(GI[k,1])-1,0]], 
				[vPrime[int(GI[k,1])-1,1]],
				[vPrime[int(GI[k,2])-1,0]], 
				[vPrime[int(GI[k,2])-1,1]],
				[vPrime[int(GI[k,3])-1,0]], 
				[vPrime[int(GI[k,3])-1,1]]])
			g = gCalc[k,:,:]
			# gCalc = np.linalg.lstsq(g.T@g,g.T, rcond=None)[0]
			t = g@v
			t_temp = (1/np.sqrt(t[0]**2 + t[1]**2)) * [[t[0], t[1]], [-t[1],t[0]]]
			T[k,:,:] = t_temp[:,:,0]
	return T


def computeVPrimePrime(edges,vertices,T,C,locations):
	VPrimePrime = np.zeros((np.size(vertices,0),2))
	A = np.zeros((np.size(edges, 0) + np.size(C), np.size(vertices, 0)))
	bx = np.zeros((np.size(edges, 0) + np.size(C),1))
	by = np.zeros((np.size(edges, 0) + np.size(C),1))
	w = 1000
	for k, edge in enumerate(edges):
		A[k,int(edge[0])-1] = -1
		A[k,int(edge[1])-1] = 1

		e = vertices[int(edge[1])-1,:] - vertices[int(edge[0])-1,:]
		Te = T[k,:,:]@e
		bx[k] = Te[0]
		by[k] = Te[1]

	for cIndex, c in enumerate(C):
		A[np.size(edges, 0)+cIndex,c] = w
		bx[np.size(edges, 0) + cIndex] = w*locations[cIndex,0]
		by[np.size(edges, 0) + cIndex] = w*locations[cIndex,1]

	Vx = np.linalg.lstsq(A.T@A,A.T@bx, rcond=None)[0]
	Vy = np.linalg.lstsq(A.T@A,A.T@by, rcond=None)[0]
	VPrimePrime[:,0] = Vx[:,0]
	VPrimePrime[:,1] = Vy[:,0]

	return VPrimePrime
