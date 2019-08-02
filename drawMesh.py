import numpy as np
import cv2
import re
import os

def read_file(file):
   for cnt, lines in enumerate(file):
      if (cnt == 0):
         no_vertices = re.findall('\d+',lines)
         no_vertices = int(no_vertices[0])
         vertices = np.zeros([no_vertices,2])

      elif (cnt == 1):
         no_faces = re.findall('\d+',lines)
         no_faces = int(no_faces[0])
         faces = np.zeros([no_faces,3])

      elif (lines[0] == 'v'):
         vertex = re.findall(r"[-+]?\d*\.*\d+", lines)
         vertex = np.array(vertex)
         vertex = vertex[:2]
         vertex = vertex.astype(np.float)
         vertices[cnt-2,:] = vertex

      elif (lines[0] == 'f'):
         face = re.findall(r"[-+]?\d*\.*\d+", lines)
         face = np.array(face)
         face = face.astype(int)
         faces[cnt-no_vertices-2,:] = face

   return no_vertices, no_faces, vertices, faces

def get_edges(no_faces,faces):
   edges = np.zeros([no_faces*3,2])
   for index, face in enumerate(faces):
      edges[index*3,:] = [face[0], face[1]]
      edges[index*3+1,:] = [face[1], face[2]]
      edges[index*3+2,:] = [face[0], face[2]]
   edges.sort(axis=1)
   edges = np.unique(edges, axis=0)
   return edges


def draw_mesh(vertices,edges,img):
   # scaling so that it fits in the window for OpenCV coordinate system
   vertices_scaled = np.zeros(np.shape(vertices))
   vertices_scaled[:,0] = vertices[:,0] * -180 + 640
   vertices_scaled[:,1] = vertices[:,1] * -180 + 400
   vertices_scaled = vertices_scaled.astype(int)
   for edge in edges:

      start = (vertices_scaled[int(edge[0]-1),0], vertices_scaled[int(edge[0]-1),1])
      end = (vertices_scaled[int(edge[1]-1),0], vertices_scaled[int(edge[1]-1),1])

      cv2.line(img, start, end, (0,255,0), 1)


def save_mesh(no_vertices, no_faces, vertices, faces, count):
   __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
   file = open(os.path.join(__location__,'man'+str(count)+'.obj'), 'w')
   file.write("#vertices: " + str(no_vertices) + "\n")
   file.write("#faces: " + str(no_faces) + "\n")
   for v in vertices:
      file.write("v " + str(v[0]) + " " + str(v[1]) + " 0\n")
   for f in faces:
      file.write("f " + str(int(f[0])) + " " + str(int(f[1])) + " " + str(int(f[2])) + "\n")
   file.close()
   print('Saved!')
