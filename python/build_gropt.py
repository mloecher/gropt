import os, sys, subprocess

def build_gropt(debug = False):
    cwd = os.getcwd()
    gropt_path = os.getcwd().rsplit('python', 1)[0] + 'python'
    imported = False

    if not gropt_path == cwd:
        print("It looks like we aren't in gropt/python/, compiling from there")
        print(gropt_path, cwd)
        os.chdir(gropt_path)

    out = subprocess.run(["python", "setup.py", "build_ext", "--inplace"], stdout=subprocess.PIPE)

    try:
        import gropt
        imported = True
    except:
        pass

    if not imported:
        print("ERROR: Couldn't build and import gropt")

    os.chdir(cwd)
    return