import sys
import site
import os

def create_module_file(python_version, install_path, output_file):
    content = f"""#%Module1.0
-- EAR job visualizer module

whatis("Enables the usage of ear-job-visualizer, a tool for visualizing performance metrics collected by EAR.")

-- Add here the required python module you used for building the package.
-- depends_on("")
prepend_path("PYTHONPATH", "{install_path}/lib/python{python_version}/site-packages")
prepend_path("PATH", "{install_path}/bin")
"""
    
    with open(output_file, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    install_path = os.environ.get('PYTHONUSERBASE', site.USER_BASE)
    output_file = "eas-tools.lua"
    
    create_module_file(python_version, install_path, output_file)