import numpy as np
import tqdm
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from openpyscad import *
from pyquaternion import Quaternion
import time

class OpenScad:

    def __init__(self, use_pytorch=True):
        self.use_pytorch = use_pytorch

    def generate_scad(self, sample_npy):
        start_time = time.time()
        np.set_printoptions(suppress=True)

        data = np.load(sample_npy, allow_pickle=True)

        primitives = data.item()['primitive'].transpose()
        boxes = primitives[:,:10]
        cylinders = primitives[:,10:18]
        spheres = primitives[:,18:26]
        cones = primitives[:,26:]

        cvx = data.item()['cvx']
        ccv = data.item()['ccv']
        cad_cubes = []
        cad_spheres = []
        cad_cylinders = []
        cad_cones = []

        for box in boxes:
            quaternion = Quaternion(box[:4]) #[w,x,y,z]
            inverse = quaternion.inverse
            quaternion = np.asarray([inverse[1], inverse[2], inverse[3],inverse[0]]) # [x,y,z,w]
            r = R.from_quat(quaternion).as_euler('xyz', degrees=True)

            translation = box[4:7]
            dim = abs(box[7:])*2
            cad_cubes.append(Cube(dim.tolist(),center=True).rotate([r[0], r[1], r[2]]).translate([translation[0], translation[1], translation[2]]))

        for cylinder in cylinders:
            quaternion = Quaternion(cylinder[:4]) #[w,x,y,z]
            inverse = quaternion.inverse
            quaternion = np.asarray([inverse[1], inverse[2], inverse[3],inverse[0]])
            r = R.from_quat(quaternion).as_euler('xyz', degrees=True)

            translation = cylinder[4:7]
            radius = abs(cylinder[-1])
            cad_cylinders.append(Cylinder(h=10, r=radius, _fn=100).translate([0,0,-5]).rotate([r[0], r[1], r[2]]).translate([translation[0], translation[1], translation[2]]))

        for sphere in spheres:
            quaternion = Quaternion(sphere[:4]) #[w,x,y,z]
            inverse = quaternion.inverse
            quaternion = np.asarray([inverse[1], inverse[2], inverse[3],inverse[0]])
            r = R.from_quat(quaternion).as_euler('xyz', degrees=True)

            translation = sphere[4:7]
            radius = abs(sphere[-1])
            cad_spheres.append(Sphere(r=radius, _fn=100).rotate([r[0], r[1], r[2]]).translate([translation[0], translation[1], translation[2]]))

        for cone in cones:
            quaternion = Quaternion(cone[:4]) #[w,x,y,z]
            inverse = quaternion.inverse
            quaternion = np.asarray([inverse[1], inverse[2], inverse[3],inverse[0]])
            r = R.from_quat(quaternion).as_euler('xyz', degrees=True)

            translation = cone[4:7]
            angle = abs(cone[-1])
            height=100
            radius_bottom = height*angle

            cad_cones.append(Cylinder(h=height, r1=0, r2=radius_bottom, _fn=100).rotate([r[0], r[1], r[2]]).translate([translation[0], translation[1], translation[2]]))

        cad_intersections = []
        cad_unions = []


        primitives = cad_cylinders + cad_cubes + cad_cones + cad_spheres
        int_nodes = []

        for i in range(cvx.shape[-1]):
            int_node = Intersection()
            for j in range(cvx.shape[0]):
                if cvx[j,i] > 0.5:
                    int_node.append(primitives[j])
            int_nodes.append(int_node)

        union = Union()
        for i in range(ccv.shape[0]):
            if ccv[i] > 0.5:
                union.append(int_nodes[i])
        union = union.scale([100, 100, 100])
        union.write("sample1.scad")

        print("Total Time: %f" % (time.time() - start_time))


    def export_array(self, file_name_prefix, start_index, batch_size, intersection_layer_connections,union_layer_connections, primitive_parameters):
        Path(file_name_prefix).mkdir(parents=True, exist_ok=True)
        
        primitive = primitive_parameters
        cvx = intersection_layer_connections
        ccv = union_layer_connections

        sample_npy = np.save('sample.npy', primitive=primitive, cvx=cvx, ccv=ccv)
        self.generate_scad(sample_npy)

