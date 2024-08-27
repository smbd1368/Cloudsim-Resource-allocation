import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import networkx as nx

# Define the XML file path
file_path = r'C:\Users\RGS\Desktop\resources\input\inputDAGfiles\inspiral300.xml'

# Define namespaces for XML parsing
namespaces = {
    '': 'http://pegasus.isi.edu/schema/DAX'
}

def parse_xml_and_create_graph(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Create a directed graph
    G = nx.DiGraph()

    # Dictionary to map job IDs to nodes
    job_to_node = {}

    # Parse jobs
    jobs = root.findall('job', namespaces)
    for job in jobs:
        job_id = job.get('id')
        name = job.get('name')
        version = job.get('version')
        runtime = float(job.get('runtime', 0))

        # Add job as a node
        label = f"{name}\nVersion: {version}\nRuntime: {runtime:.2f}"
        G.add_node(job_id, label=label)
        job_to_node[job_id] = job

    # Parse child elements for dependencies
    children = root.findall('child', namespaces)
    for child in children:
        child_id = child.get('ref')
        parents = child.findall('parent', namespaces)
        for parent in parents:
            parent_id = parent.get('ref')
            if parent_id in job_to_node and child_id in job_to_node:
                G.add_edge(parent_id, child_id)

    return G

def plot_tree_graph(G, output_file):
    # Use a tree layout for the graph
    pos = nx.planar_layout(G)  # Use planar layout for a tree-like appearance

    # Extract labels for nodes
    labels = nx.get_node_attributes(G, 'label')

    # Draw the graph
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, arrowsize=20, edge_color='gray')
    
    # Set the title and save the figure
    plt.title('Workflow Task Dependencies (Tree Layout)')
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")

def main():
    # Create the graph from XML
    G = parse_xml_and_create_graph(file_path)
    
    # Plot and save the graph
    output_file = 'workflow_dependencies_tree.png'
    plot_tree_graph(G, output_file)

if __name__ == "__main__":
    main()
