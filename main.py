import numpy as np
import cv2
import drawMesh
import stepOne
import stepTwo
import os

# some global variables
startSelect = False
endSelect = False
startMove = False
endMove = False
moving = -1
selected_old = [-1]
A = 0
b = 0
moving_old = -1
started = False
new_vertices = 0

#This funtction is just to see if there is a vertex near where you click so that it can be selected 
def isClose(x,y,vertices):
	vertices_scaled = vertices.copy()
	vertices_scaled[:,0] = vertices[:,0] * -180 + 640
	vertices_scaled[:,1] = vertices[:,1] * -180 + 400
	vertices_scaled = vertices_scaled.astype(int)
	close = False
	for i in range(x-4,x+4):
		for j in range(y-4, y+4):
			if ([i,j] == vertices_scaled).all(axis=1).any():
				close = True
				click = np.array([i,j])
				index = np.where(np.all(click==vertices_scaled,axis=1))
				index = index[0][0]
				break
	if not close:
		index = -1

	return close,index

# mouse events
def move_mesh(event, x, y, flags, param):
	global x1, y1, img, img2, verticesSelect, verticesMove, vertices, edges, moving, gCalc, GI, H, selected_old, A, b, moving_old, started, new_vertices					

	if event == cv2.EVENT_LBUTTONDOWN:
		close,index = isClose(x,y,vertices)
		if started:
			close,index = isClose(x,y,new_vertices)
		if close:
			if verticesSelect[index] == 0:
				x1, y1 = x, y
				cv2.circle(img, (x1,y1), 5, (0, 0, 255), 1)
				verticesSelect[index] = 1

			elif verticesSelect[index] == 1:
				verticesMove[index] = 1
				moving = index
				if not started:
					moving_old = moving   

	elif event == cv2.EVENT_MOUSEMOVE:
    	# close,index = isClose(x,y,vertices)
		if moving >= 0:
			if verticesMove[moving] == 1:
				a, b = x, y
				if a != x & b != y:
					img = img2.copy()
					selected = np.where(verticesSelect==1)
					selected = selected[0]

					# update the locations of the selected points
					if not started:
						locations = vertices[selected,:]
					else:
						locations = new_vertices[selected,:]

					a = (a - 640) / -180 
					b = (b - 400) / -180 
					location = [a,b]
					movingI = np.where(selected==moving)
					movingI = movingI[0][0]

					# locations of the selected points
					locations[movingI,:] = location

		        	# /////// MAIN CALCULATIONS HERE //////////////

					if not np.array_equal(selected,selected_old):
						new_vertices, A, b = stepOne.computeVPrime(edges, vertices, GI, H, selected, locations)
						T = stepTwo.computeT(edges, gCalc, GI, new_vertices)
						new_vertices = stepTwo.computeVPrimePrime(edges, vertices, T, selected, locations)
						started = True
					else:
						# just reusing A and b so they dont have to be recalculated
						new_vertices = stepOne.computeVPrimeFast(edges, vertices, selected, locations, A, b)
						T = stepTwo.computeT(edges, gCalc, GI, new_vertices)
						new_vertices = stepTwo.computeVPrimePrime(edges, vertices, T, selected, locations)
						selected_old = selected

					drawMesh.draw_mesh(new_vertices,edges,img)

					new_vertices_scaled = new_vertices.copy()
					new_vertices_scaled[:,0] = new_vertices_scaled[:,0] * -180 + 640
					new_vertices_scaled[:,1] =new_vertices_scaled[:,1] * -180 + 400
					new_vertices_scaled = new_vertices_scaled.astype(int)
					for i in selected:
						cv2.circle(img, (new_vertices_scaled[i,0],new_vertices_scaled[i,1]), 5, (0, 0, 255), 1)


	elif event == cv2.EVENT_LBUTTONUP:
		if verticesMove[moving] == 1:
			verticesMove[moving] = 0
			moving = -1

		font = cv2.FONT_HERSHEY_SIMPLEX


if __name__ == "__main__":
	num = 0
	windowName = 'ARAP'
	cv2.setUseOptimized(True)

	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

	yofile_object  = open(os.path.join(__location__,'man.obj'), 'r') 
	no_vertices, no_faces, vertices, faces = drawMesh.read_file(yofile_object)
	verticesMove = np.zeros(no_vertices+1)
	verticesSelect = np.zeros(no_vertices+1)

	img = np.zeros((800, 1280, 3), np.uint8)
	img2 = img.copy()
	edges = drawMesh.get_edges(no_faces,faces)

	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, move_mesh)

	GI, gCalc = stepOne.computeG(vertices, edges, faces)
	H = stepOne.computeH(edges, gCalc, GI,vertices)

	count = 0
	drawMesh.draw_mesh(vertices,edges,img)
	cv2.namedWindow(windowName)
	while (True):

		cv2.imshow(windowName, img)
		# press space bar to save an .obj file of the deformed mesh
		if cv2.waitKey(1) == 32:
			drawMesh.save_mesh(no_vertices, no_faces, new_vertices, faces, count)
			count +=1

	cv2.destroyAllWindows()
