import numpy as np
import cv2

def computeRotationsAndTransforms(V_start, V_end, faces, no_vertices, no_faces):
	A_transforms = np.zeros((no_faces,4,4))
	P_inverse = np.zeros((no_vertices,6,6))
	R = np.zeros((no_faces,4,4))
	S = np.zeros((no_faces,4,4))
	W = np.zeros(no_faces)
	quaternions = np.zeros((no_faces,4))

	Q_0 = [1,0,0,0]
	# Q_1 =  np.zeros((no_vertices,2))

	W = np.zeros(no_faces)

	for i, face in enumerate(faces):

		P = [[V_start[face[0],0], V_start[face[0],1],1,0,0,0],
			[0,0,0,V_start[face[0],0], V_start[face[0],1],1],
			[V_start[face[1],0], V_start[face[1],1],1,0,0,0],
			[0,0,0,V_start[face[1],0], V_start[face[1],1],1],
			[V_start[face[2],0], V_start[face[2],1],1,0,0,0],
			[0,0,0,V_start[face[2],0], V_start[face[2],1],1]]


		Q = [V_end[face[0],0],
			V_end[face[0],1],
			V_end[face[1],0],
			V_end[face[1],1],
			V_end[face[2],0],
			V_end[face[2],1]]

		a =  np.linalg.lstsq(P,Q, rcond=None)[0]

		A_transform = [[a[0], a[1]],
						[a[3],a[4]]]

		R_a, D, R_bt = np.linalg.svd(A_transform, full_matrices=True)
		R_b = R_bt.T

		rotation = R_a@R_b
		symmetry = R_b.T@D@R_b

		 # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
		 # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
		 # http://www.cs.cmu.edu/afs/cs/academic/class/16741-s07/www/lectures/Lecture8.pdf

		q0 = sqrt(2+rotation[0,0]+rotation[1,1])/2
		q3 = (rotation[1,0] - rotation[0,1])/(4*q0)

		Q_1 = [q0, 0, 0, q3]

		w = np.arccos(Q_0 * Q_1)

		P_inverse[i,:,:] = np.linalg.inv(P)
		A_transforms[i,:,:] = A_transform
		R[i,:,:] = rotation
		S[i,:,:] = symmetry
		W[i] = w
		quaternions = Q_1

	return P_inverse, A_transforms, R, S, W, quaternions

def computVt(P_inverse, A_transforms, R, S, W, quaternions, t, no_faces, no_vertices):

	Q_0 = [1,0,0,0]

	I = [[1,0],
		[0,1]]

	A = np.zeros((4*no_faces+2, 2*no_vertices))
	b = np.zeros(4*no_faces+2)

	V_t = np.zeros((no_vertices,2))

	for i, face in enumerate(faces):
		Q_t = (Q_0*np.sin((1-t)*W[i]) + quaternions[i,:]*np.sin(t*W[i]))/np.sin(W[i])

		R_t = [[1-2*(Q_t[2]**2 +Q_t[3]**2), 2*(Q_t[1]*Q_t[2] - Q_t[0]*Q_t[3])],
			[2*(Q_t[1]*Q_t[2] + Q_t[0]*Q_t[3]), 1-2*(Q_t[1]**2 +Q_t[3]**2)]]

		A_t = R_t*((1-t)*I + t*S[i,:,:])

		b[i*4] = A_t[0,0]
		b[i*4+1] = A_t[0,1]
		b[i*4+2] = A_t[1,0]
		b[i*4+3] = A_t[1,1]

		for j, vertex in enumerate(face):
			A[i*4, vertex*2] = P_inverse[i,0,0]
			A[i*4, vertex*2+1] = P_inverse[i,0,1]
			A[i*4+1, vertex*2] = P_inverse[i,1,0]
			A[i*4+1, vertex*2+1] = P_inverse[i,1,1]
			A[i*4+2, vertex*2] = P_inverse[i,3,0]
			A[i*4+2, vertex*2+1] = P_inverse[i,3,1]
			A[i*4+3, vertex*2] = P_inverse[i,4,0]
			A[i*4+3, vertex*2+1] = P_inverse[i,4,1]

		V = np.linalg.lstsq(A.T@A,A.T@b, rcond=None)[0]
		V = [:-2]

		V_t[:,0] = V[0::2,0]
		V_t[:,1] = V[1::2,0]










