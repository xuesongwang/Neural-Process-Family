import subprocess
from os import rename, listdir


def rename_file(kernel):
    fnames = listdir("CNP/"+kernel)
    for fname in fnames:
        print (fname)
        rename('CNP/'+kernel+'/'+fname,'CNP/'+kernel+'/'+ "%03d"%(int(fname[:-4])) +".png")

def build_gif(modelname, kernel):
    # Build GIF.
    subprocess.call(['convert',
                     '-delay', '20',
                     '-loop', '0',
                     modelname +'/'+kernel+'/*.png', +modelname+'_'+ kernel+'.gif'])
    print("finished GIF!")

if __name__ == '__main__':
    kernel = 'period'
    MODELNAME = 'ConvCNP'
    # rename_file(kernel)
    build_gif(MODELNAME, kernel)



