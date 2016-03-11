import numpy
import sys
import os

def get_mesh_filename(srf_file, mesh_folder):
    mesh_files = numpy.array([])
    mesh_type  = numpy.array([])
    for line in file(srf_file):
        line = line.split()

        for word in line:
            if 'surf' in word:
                i = word.find('surf')
                mesh_files = numpy.append(mesh_files, mesh_folder+word[i+1:])
                if len(line)>1:
                    mesh_type = numpy.append(mesh_type, 'internal_cavity')
                elif len(line)==1:
                    if len(mesh_files)>0:
                        mesh_type = numpy.append(mesh_type, 'dielectric_interface')
                    elif len(mesh_files)==0:
                        meh_type = numpy.append(mesh_type, 'stern_layer')

    return mesh_files, mesh_type


def write_config_file(config_filename, mesh_files, mesh_type, pqr_file):
    
    
    


srf_file = sys.argv[1]
mesh_folder = sys.argv[2]
pqr_folder = sys.argv[3]

if mesh_folder[-1] != '/':
    mesh_folder += '/'

mesh_files, mesh_type = get_mesh_file(srf_file, mesh_folder)

for dir_path, dir_names, file_names in os.walk(pqr_folder):

    if file_names[-4:] == '.pqr':
        
