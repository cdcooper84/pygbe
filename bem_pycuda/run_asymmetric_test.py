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
                mesh_files = numpy.append(mesh_files, mesh_folder+word[i:])
                if len(line)>1:
                    mesh_type = numpy.append(mesh_type, 'internal_cavity')
                elif len(line)==1:
                    if len(mesh_type)>0:
                        mesh_type = numpy.append(mesh_type, 'dielectric_interface')
                    elif len(mesh_type)==0:
                        mesh_type = numpy.append(mesh_type, 'stern_layer')

    return mesh_files, mesh_type

def find_pqr_files(pqr_folder):

    pqr_files = numpy.array([])
    for file_name in os.listdir(pqr_folder):
        if file_name.endswith('.pqr'):
                pqr_files = numpy.append(pqr_files, pqr_folder+file_name)
                    
    return pqr_files


def write_config_file(config_filename, mesh_files, mesh_type, pqr_file):

    f = open(config_filename, 'w')

#   Write FILE lines
    for i in range(len(mesh_files)):
        line_string = 'FILE\t'+mesh_files[i]+'\t'+mesh_type[i]+'\n'
        f.write(line_string)

#   Write intermediate lines
    f.write('--------------------------------\n')
    f.write('PARAM\tLorY\tE?\tDielec\tkappa\tcharges?\tcoulomb?\tcharge_file\tNparent\tparent\tNchild\tchildren\n')

#   Write FIELD lines
    Ncav = len(numpy.where(mesh_type=='internal_cavity')[0])
#   Outside fields
    for i in range(len(mesh_files)):
        if mesh_type[i]=='stern_layer':
            LorY = '2'; E='0'; dielec='80'; kappa='0.125'; charges='0'; coulomb='0'; charge_file='NA'; Nparent='0'; parent='NA'; Nchild='1'; children=str(i)
        elif mesh_type[i]=='dielectric_interface':
            LorY = '1'; E='0'; dielec='80'; kappa='1e-12'; charges='0'; coulomb='0'; charge_file='NA'; Nparent='1'; parent=str(i-1); Nchild='1'; children=str(i)
        elif mesh_type[i]=='internal_cavity':
            LorY = '1'; E='1'; dielec='4'; kappa='1e-12'; charges='1'; coulomb='1'; charge_file=pqr_file; Nparent='1'; parent=str(i-1); Nchild=str(Ncav); children=str(numpy.arange(i,i+Ncav, dtype=int))[1:-1]
        else:
            break
    
        line_string = 'FIELD\t'+LorY+'\t'+E+'\t'+dielec+'\t'+kappa+'\t'+charges+'\t'+coulomb+'\t'+charge_file+'\t'+Nparent+'\t'+parent+'\t'+Nchild+'\t'+children+'\n'

        f.write(line_string)

#   Fields inside cavities 
    Nsurf = len(mesh_type)
    i_cav = numpy.where(mesh_type=='internal_cavity')[0]
    if len(i_cav)==0:
        LorY = '1'; E='1'; dielec='4'; kappa='1e-12'; charges='1'; coulomb='1'; charge_file=pqr_file; Nparent='1'; parent='1'; Nchild='0'; children='NA'
        line_string = 'FIELD\t'+LorY+'\t'+E+'\t'+dielec+'\t'+kappa+'\t'+charges+'\t'+coulomb+'\t'+charge_file+'\t'+Nparent+'\t'+parent+'\t'+Nchild+'\t'+children+'\n'
        f.write(line_string)
    else:
        for i in i_cav:
            LorY = '1'; E='0'; dielec='80'; kappa='1e-12'; charges='0'; coulomb='0'; charge_file='NA'; Nparent='1'; parent=str(i); Nchild='0'; children='NA'
            line_string = 'FIELD\t'+LorY+'\t'+E+'\t'+dielec+'\t'+kappa+'\t'+charges+'\t'+coulomb+'\t'+charge_file+'\t'+Nparent+'\t'+parent+'\t'+Nchild+'\t'+children+'\n'
            f.write(line_string)

    f.close()

def scanOutput(filename):
    
    flag = 0
    for line in file(filename):
        line = line.split()
        if len(line)>0:
            if line[0]=='Converged':
                iterations = int(line[2])
            if line[0]=='Total' and line[1]=='elements':
                N = int(line[-1])
            if line[0]=='Totals:':
                flag = 1
            if line[0]=='Esolv' and flag==1:
                Esolv = float(line[2])
            if line[0]=='Esurf' and flag==1:
                Esurf = float(line[2])
            if line[0]=='Ecoul' and flag==1:
                Ecoul = float(line[2])
            if line[0]=='Time' and flag==1:
                Time = float(line[2])

    return N, iterations, Esolv, Esurf, Ecoul, Time

srf_file = sys.argv[1]
mesh_folder = sys.argv[2]
pqr_folder = sys.argv[3]
param_file = sys.argv[4]

if mesh_folder[-1] != '/':
    mesh_folder += '/'

# Generate config files
mesh_files, mesh_type = get_mesh_filename(srf_file, mesh_folder)
pqr_files_array = find_pqr_files(pqr_folder)
config_filename_array = numpy.array([])
for pqr_file in pqr_files_array:
    i1 = pqr_file.find('.')
    i2 = srf_file.find('.')

    config_filename = srf_file[:i2]+'_'+pqr_file[i1-3:i1]+'.config'
    config_filename_array = numpy.append(config_filename_array, config_filename)

    write_config_file(config_filename, mesh_files, mesh_type, pqr_file)

# Run cases
print 'ASYMMETRIC RUNS'
Esolv_asym = numpy.zeros(len(config_filename_array)) 
for i, config_filename in enumerate(config_filename_array):
    
    pqr_file = pqr_files_array[i]

    i1 = pqr_file.find('.')
    i2 = srf_file.find('.')

    output_file = 'output_asym_'+srf_file[i2-6:i2]+'_'+pqr_file[i1-3:i1]
    command = './main_asymmetric.py '+param_file+' '+config_filename+' --asymmetric --chargeForm >'+output_file

    print 'Running '+config_filename+', saved on '+output_file+'...'

    os.system(command)
    N,iterations,Esolv_asym[i],Esurf,Ecoul,Time = scanOutput(output_file)

print 'Summary'
print 'Surface\tpqr\tEsolv'
print '----------------------------------------------'
for i, config_filename in enumerate(config_filename_array):
    
    pqr_file = pqr_files_array[i]

    i1 = pqr_file.find('.')
    i2 = srf_file.find('.')

    print srf_file[i2-6:i2]+'\t'+pqr_file[i1-3:i1]+'\t'+str(Esolv_asym[i])

print 'SYMMETRIC RUNS'
Esolv_sym = numpy.zeros(len(config_filename_array)) 
for i, config_filename in enumerate(config_filename_array):
    
    pqr_file = pqr_files_array[i]

    i1 = pqr_file.find('.')
    i2 = srf_file.find('.')

    output_file = 'output_sym_'+srf_file[i2-6:i2]+'_'+pqr_file[i1-3:i1]
    command = './main_asymmetric.py '+param_file+' '+config_filename+'>'+output_file

    print 'Running '+config_filename+', saved on '+output_file+'...'

    os.system(command)
    N,iterations,Esolv_sym[i],Esurf,Ecoul,Time = scanOutput(output_file)

print 'Summary'
print 'Surface\tpqr\tEsolv'
print '----------------------------------------------'
for i, config_filename in enumerate(config_filename_array):
    
    pqr_file = pqr_files_array[i]

    i1 = pqr_file.find('.')
    i2 = srf_file.find('.')

    print srf_file[i2-6:i2]+'\t'+pqr_file[i1-3:i1]+'\t'+str(Esolv_sym[i])
