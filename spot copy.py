import xml.etree.ElementTree as ET

# Path to the XML file
file_path = r'C:\Users\RGS\Desktop\resources\input\inputDAGfiles\inspiral300.xml'

# Parse the XML file
tree = ET.parse(file_path)
root = tree.getroot()

# Define namespaces
namespaces = {
    '': 'http://pegasus.isi.edu/schema/DAX'
}

# Find all job elements
jobs = root.findall('job', namespaces)

# Iterate over job elements and print details
for job in jobs:
    job_id = job.get('id')
    name = job.get('name')
    version = job.get('version')
    runtime = job.get('runtime')
    
    print(f"Job ID: {job_id}")
    print(f"Name: {name}")
    print(f"Version: {version}")
    print(f"Runtime: {runtime}")
    
    print("Uses:")
    for uses in job.findall('uses', namespaces):
        file_name = uses.get('file')
        link = uses.get('link')
        size = uses.get('size')
        print(f"  File: {file_name}, Link: {link}, Size: {size}")
    
    print("-" * 40)
