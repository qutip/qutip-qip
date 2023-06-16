#Adapted from https://github.com/qutip/qutip-tutorials/blob/1d1da2fc623372fa11ad5370a4fcd19452aad8fa/website/create_index.py

import os
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape
import shutil
import subprocess
import tempfile

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

class notebook:
    def __init__(self, path, title):
        # remove ../ from path
        self.path = path.replace('../', '')
        self.title = title
        # set url and update from markdown to ipynb
        self.url = url_prefix + self.path.replace(".md", ".ipynb")
        self.url=self.url.replace(cloned_repo_dir,"")
        
def get_title(filename):
    """ Reads the title from a markdown notebook """
    with open(filename, 'r') as f:
        # get first row that starts with "# "
        for line in f.readlines():
            # trim leading/trailing whitespaces
            line = line.lstrip().rstrip()
            # check if line is the title
            if line[0:2] == '# ':
                # return title
                return line[2:]

def sort_files_titles(files, titles):
    """ Sorts the files and titles either by filenames or titles """
    # identify numbered files and sort them
    nfiles = [s for s in files if s.split('/')[-1][0].isdigit()]
    nfiles = sorted(nfiles, key=natural_keys)
    ntitles = [titles[files.index(s)] for s in nfiles]
    # sort the files without numbering by the alphabetic order of the titles
    atitles = [titles[files.index(s)] for s in files if s not in nfiles]
    atitles = sorted(atitles, key=natural_keys)
    afiles = [files[titles.index(s)] for s in atitles]
    # merge the numbered and unnumbered sorting
    return nfiles + afiles, ntitles + atitles

def get_notebooks(path):
    """ Gets a list of all notebooks in a directory """
    # get list of files and their titles
    try:
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.md')]
    except FileNotFoundError:
        return {}
    titles = [get_title(f) for f in files]
    # sort the files and titles for display
    files_sorted, titles_sorted = sort_files_titles(files, titles)
    # generate notebook objects from the sorted lists and return
    notebooks = [notebook(f, t) for f, t in zip(files_sorted, titles_sorted)]
    return notebooks

def generate_index_html(version_directory, tutorial_directories, title, version_note):
    """ Generates the index HTML file from the given data """
    # get tutorials from the different directories
    tutorials = {}
    for dir in tutorial_directories:
        tutorials[dir] = get_notebooks(os.path.join(version_directory, dir))
    
    # Load environment for Jinja and template
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../")),
        autoescape=select_autoescape()
    )
    template = env.get_template("tutorials-website/tutorials.html.jinja")

    # render template and return
    html = template.render(tutorials=tutorials, title=title, version_note=version_note)
    return html

# Clone the qutip-tutorials repository
repo_url = 'https://github.com/qutip/qutip-tutorials.git'
cloned_repo_dir = tempfile.mkdtemp()
subprocess.run(['git', 'clone', repo_url, cloned_repo_dir])

# Set the necessary variables
url_prefix = "https://nbviewer.org/urls/qutip.org/qutip-tutorials"
tutorial_directories = [
    'pulse-level-circuit-simulation',
    'quantum-circuits',
]

# Perform the operations on the cloned repository
prefix = ""
suffix = ""
#with open('prefix.html', 'r') as f:
 #   prefix = f.read()
#with open('suffix.html', 'r') as f:
#    suffix = f.read()

# Version 4 index file
title = 'Tutorials for QuTiP Version 4'
version_note = 'These are the tutorials for QuTiP Version 4. You can find the tutorials for QuTiP Version 5 <a href="./index-v5.html">here</a>.'
html = generate_index_html(os.path.join(cloned_repo_dir, 'tutorials-v4/'), tutorial_directories, title, version_note)
with open('source/tutorials-website/qutip-qip.html', 'w+') as f:
    #f.write(prefix)
    f.write(html)
    #f.write(suffix)

# Version 5 index file
title = 'Tutorials for QuTiP Version 5'
version_note = 'These are the tutorials for QuTiP Version 5. You can find the tutorials for QuTiP Version 4 <a href="./index.html">here</a>.'
html = generate_index_html(os.path.join(cloned_repo_dir, 'tutorials-v5/'), tutorial_directories, title, version_note)
with open('source/tutorials-website/qutip-qip-v5.html', 'w+') as f:
    #f.write(prefix)
    f.write(html)
    #f.write(suffix)

# Wipe off the cloned repository
shutil.rmtree(cloned_repo_dir)

def convert_html_to_rst(html_file_path, rst_file_path):
     # Use the subprocess module to call the pandoc command-line tool
     subprocess.run(['pandoc', html_file_path, '-o', rst_file_path])

html_file_path = 'source/tutorials-website/qutip-qip.html'
html_file_path_v5 = 'source/tutorials-website/qutip-qip-v5.html'

rst_file_path = 'source/tutorials.rst'
rst_file_path_v5 = 'source/tutorials_v5.rst'

#convert_html_to_rst(html_file_path, rst_file_path)
convert_html_to_rst(html_file_path_v5, rst_file_path_v5)


