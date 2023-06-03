 
import base64
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape
import requests
import subprocess


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


class notebook:
    def __init__(self, path, title):
        # remove ../ from path
        self.path = path.replace('https://github.com/qutip/qutip-tutorials/tree/main/', '')
        self.title = title
        # set url and update from markdown to ipynb
        self.url = url_prefix + self.path.replace(".md", ".ipynb")


def get_title(filename, dir, version):
    """ Reads the title from the notebook """
    file_url = f"https://api.github.com/repos/qutip/qutip-tutorials/contents/{version}/{dir}/{filename}"
    response = requests.get(file_url)
    file_content = response.json()['content']
    decoded_text= base64.b64decode(file_content).decode('utf-8') 
    lines = decoded_text.split("\n")
    for line in lines:
    # trim leading/trailing whitespaces
        line = line.strip()
        # check if line is the title
        if line.startswith('#'):
            # return title
            title = line[2:]
            return title


def get_directory_contents(directory, version):
    url = f"https://api.github.com/repos/qutip/qutip-tutorials/contents/{version}/{directory}"
    response = requests.get(url)
    if response.status_code == 200:
        contents = response.json()
        file_names = [content['name'] for content in contents if content['type'] == 'file']
        return file_names
    
    else:
        print(response.status_code)
        return None

def get_notebooks(path,version,dir):
    """ Gets a list of all notebooks in a directory """
    # get list of files and their titles
    try:
        filesintut = get_directory_contents(dir,version)
        files = [path + f for f in filesintut if f.endswith('.md') ] 
    except FileNotFoundError:
        return {}
    titles = [get_title(f,dir,version) for f in filesintut]
    # sort the files and titles for display: TODO : files_sorted, titles_sorted = sort_files_titles(files, titles)
    # generate notebook objects from the lists and return
    notebooks = [notebook(f, t) for f, t in zip(files, titles)]
    return notebooks


def generate_index_html(version_directory, version, tutorial_directories, title,
                        version_note):
    """ Generates the index html file from the given data"""
    # get tutorials from the different directories
    tutorials = {}
    for dir in tutorial_directories:
        tutorials[dir] = get_notebooks(version_directory + dir + '/', version, dir)

    # Load environment for Jinja and template
    env = Environment(
        loader=FileSystemLoader("../"),
        autoescape=select_autoescape()
    )
    template = env.get_template("tutorials-website/tutorials.html.jinja")

    # render template and return
    html = template.render(tutorials=tutorials, title=title,
                           version_note=version_note)
    return html


# url prefix for the links
url_prefix = "https://nbviewer.org/urls/qutip.org/qutip-tutorials/"
# tutorial directories
tutorial_directories = [
    'quantum-circuits',
     'pulse-level-circuit-simulation',
]

# +++ READ PREFIX AND SUFFIX +++
prefix = ""
suffix = ""

#with open('prefix.html', 'r') as f:
#   prefix = f.read()
#with open('suffix.html', 'r') as f:
#    suffix = f.read()

# +++ VERSION 4 INDEX FILE +++
title = 'Tutorials'
version_note = 'This are the tutorials for QuTiP-qip Version 4. You can \
         find the tutorials for QuTiP Version 5 \
          <a href="./qutip-qip-v5.html">here</a>.'
version_directory= 'https://github.com/qutip/qutip-tutorials/tree/main/tutorials-v4/'
html = generate_index_html('https://github.com/qutip/qutip-tutorials/tree/main/tutorials-v4/', "tutorials-v4", tutorial_directories, title,version_note)
with open('qutip-qip.html', 'w+') as f:
    f.write(prefix)
    f.write(html)
    f.write(suffix)

# +++ VERSION 5 INDEX FILE +++
title = 'Tutorials'
version_note = 'This are the tutorials for QuTiP Version 5. You can \
         find the tutorials for QuTiP Version 4 \
          <a href="./qutip-qip.html">here</a>.'

html = generate_index_html('https://github.com/qutip/qutip-tutorials/tree/main/tutorials-v5/', "tutorials-v5",tutorial_directories, title,version_note)
with open('qutip-qip-v5.html', 'w+') as f:
    f.write(prefix)
    f.write(html)
    f.write(suffix)

# convert html to rst
def convert_html_to_rst(html_file_path, rst_file_path):
     # Use the subprocess module to call the pandoc command-line tool
     subprocess.run(['pandoc', html_file_path, '-o', rst_file_path])

html_file_path = 'qutip-qip.html'
html_file_path_v5 = 'qutip-qip-v5.html'

rst_file_path = '../tutorials.rst'
rst_file_path_v5 = '../tutorials_v5.rst'

convert_html_to_rst(html_file_path, rst_file_path)
convert_html_to_rst(html_file_path_v5, rst_file_path_v5)

