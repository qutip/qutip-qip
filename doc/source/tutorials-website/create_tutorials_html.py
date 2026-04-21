# Adapted from https://github.com/qutip/qutip-tutorials/blob/1d1da2fc623372fa11ad5370a4fcd19452aad8fa/website/create_index.py

import os
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape
import shutil
import subprocess
import tempfile


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class notebook:
    def __init__(self, path, title):
        # remove ../ from path
        self.path = path.replace("../", "")
        self.title = title
        # set url and update from markdown to ipynb
        self.url = url_prefix + self.path.replace(".md", ".ipynb")
        self.url = self.url.replace(cloned_repo_dir, "")


def get_title(filename):
    """Reads the title from a markdown notebook"""
    with open(filename, "r") as f:
        # get first row that starts with "# "
        for line in f.readlines():
            # trim leading/trailing whitespaces
            line = line.lstrip().rstrip()
            # check if line is the title
            if line[0:2] == "# ":
                # return title
                return line[2:]


def sort_files_titles(files, titles):
    """Sorts the files and titles either by filenames or titles"""
    # identify numbered files and sort them
    nfiles = [s for s in files if s.split("/")[-1][0].isdigit()]
    nfiles = sorted(nfiles, key=natural_keys)
    ntitles = [titles[files.index(s)] for s in nfiles]
    # sort the files without numbering by the alphabetic order of the titles
    atitles = [titles[files.index(s)] for s in files if s not in nfiles]
    atitles = sorted(atitles, key=natural_keys)
    afiles = [files[titles.index(s)] for s in atitles]
    # merge the numbered and unnumbered sorting
    return nfiles + afiles, ntitles + atitles


def get_notebooks(path):
    """Gets a list of all notebooks in a directory"""
    # get list of files and their titles
    try:
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".md")]
    except FileNotFoundError:
        return {}
    titles = [get_title(f) for f in files]
    # sort the files and titles for display
    files_sorted, titles_sorted = sort_files_titles(files, titles)
    # generate notebook objects from the sorted lists and return
    notebooks = [notebook(f, t) for f, t in zip(files_sorted, titles_sorted)]
    return notebooks


def generate_index_rst(version_directory, tutorial_directories, title):
    """Generates the index RST file natively from the given data"""
    tutorials = {}
    for dir in tutorial_directories:
        tutorials[dir] = get_notebooks(os.path.join(version_directory, dir))

    # Load environment for Jinja and template
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../")),
        autoescape=select_autoescape(),
    )

    template = env.get_template("tutorials-website/tutorials.rst.jinja")

    # render template and return
    rst_content = template.render(
        tutorials=tutorials, title=title
    )  # Removed version_note
    return rst_content


# Clone the qutip-tutorials repository
repo_url = "https://github.com/qutip/qutip-tutorials.git"
cloned_repo_dir = tempfile.mkdtemp()
subprocess.run(["git", "clone", repo_url, cloned_repo_dir])

# Set the necessary variables
url_prefix = "https://nbviewer.org/urls/qutip.org/qutip-tutorials"
tutorial_directories = [
    "pulse-level-circuit-simulation",
    "quantum-circuits",
]

# Version 5 index file
title = "Tutorials for QuTiP Version 5"
rst_content = generate_index_rst(
    os.path.join(cloned_repo_dir, "tutorials-v5/"),
    tutorial_directories,
    title,
)
with open("source/tutorials-website/tutorials_v5.rst", "w+") as f:
    f.write(rst_content)

# Wipe off the cloned repository
shutil.rmtree(cloned_repo_dir)
