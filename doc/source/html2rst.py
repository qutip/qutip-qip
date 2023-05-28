import subprocess

def convert_html_to_rst(html_file_path, rst_file_path):
    # Use the subprocess module to call the pandoc command-line tool
    subprocess.run(['pandoc', html_file_path, '-o', rst_file_path])

html_file_path = 'qutip-qip.html'
rst_file_path = 'tutorials.rst'
convert_html_to_rst(html_file_path, rst_file_path)