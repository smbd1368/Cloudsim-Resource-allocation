import xml.etree.ElementTree as ET
import random
import os
import time
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from enum import Enum

# Define ResourceType Enum
class ResourceType(Enum):
    CPU = 111
    GPU = 22
    FPGA = 31

# Define InstanceType class
class InstanceType:
    def __init__(self, id, cpu, gpu, fpga, cost_per_hour):
        self.id = id
        self.resources = {
            ResourceType.CPU: cpu,
            ResourceType.GPU: gpu,
            ResourceType.FPGA: fpga
        }
        self.cost_per_hour = cost_per_hour

# Define Region class
class Region:
    def __init__(self, name):
        self.name = name
        self.datacenters = []

    def add_datacenter(self, datacenter):
        self.datacenters.append(datacenter)

# Define DataCenter class
class DataCenter:
    def __init__(self, name):
        self.name = name
        self.hosts = []

    def add_host(self, host):
        self.hosts.append(host)

    def get_total_resources(self):
        total_resources = defaultdict(int)
        for host in self.hosts:
            for resource_type, amount in host.resources.items():
                total_resources[resource_type] += amount
        return total_resources

# Define Host class
class Host:
    def __init__(self, id, resources):
        self.id = id
        self.resources = resources
        self.vms = []

    def add_vm(self, vm):
        if self.can_host(vm):
            self.vms.append(vm)
            self.allocate_resources(vm)
            return True
        return False

    def can_host(self, vm):
        return all(self.resources[r] >= vm.resources[r] for r in ResourceType)

    def allocate_resources(self, vm):
        for r in ResourceType:
            self.resources[r] -= vm.resources[r]

    def release_resources(self, vm):
        for r in ResourceType:
            self.resources[r] += vm.resources[r]

# Define VM class
class VM:
    def __init__(self, id, resources):
        self.id = id
        self.resources = resources
        self.tasks = deque()

    def add_task(self, task):
        if self.can_host(task):
            self.tasks.append(task)
            return True
        return False

    def can_host(self, task):
        return all(self.resources[r] >= task.resource_requirements[r] for r in ResourceType)

# Define Task class
class Task:
    def __init__(self, id, resource_requirements, length, priority=0):
        self.id = id
        self.resource_requirements = resource_requirements
        self.length = length  # Original length as an arbitrary unit
        self.dependencies = set()
        self.vm = None  # To track which VM executed this task
        self.priority = priority
        self.runtime = 0  # To record the runtime of the task
        self.cost = 0  # To record the cost of the task

    def add_dependency(self, task):
        self.dependencies.add(task)

    def __repr__(self):
        return f"Task(id={self.id}, priority={self.priority})"

    def start(self, instance_type):
        # Start the task and record start time
        self.start_time = time.time()

        # Compute runtime based on resource requirements and instance type
        total_available_resources = (
            instance_type.resources[ResourceType.CPU] * ResourceType.CPU.value +
            instance_type.resources[ResourceType.GPU] * ResourceType.GPU.value +
            instance_type.resources[ResourceType.FPGA] * ResourceType.FPGA.value
        )
        required_resources = (
            self.resource_requirements[ResourceType.CPU] * ResourceType.CPU.value +
            self.resource_requirements[ResourceType.GPU] * ResourceType.GPU.value +
            self.resource_requirements[ResourceType.FPGA] * ResourceType.FPGA.value
        )
        
        self.runtime = (self.length / required_resources) * total_available_resources
        # Simulate task runtime
        # time.sleep(self.runtime)

    def end(self):
        # End the task and record end time
        self.end_time = time.time()
        # Calculate cost based on runtime and cost per hour of the instance type
        if self.vm:
            self.cost = (self.runtime / 3600) * self.vm.resources[ResourceType.CPU] * 0.01  # Example cost factor

# Define Workflow class
class Workflow:
    def __init__(self, id):
        self.id = id
        self.tasks = []
        self.task_dict = {}  # To quickly lookup tasks by their id
        self.total_runtime = 0  # To track total runtime of the workflow

    def add_task(self, task):
        self.tasks.append(task)
        self.task_dict[task.id] = task

    def add_dependency(self, task_id, dependency_id):
        task = self.task_dict.get(task_id)
        dependency = self.task_dict.get(dependency_id)
        if task and dependency:
            task.add_dependency(dependency)

    def get_dependencies(self, task):
        return [dep for dep in self.tasks if task in dep.dependencies]

    def __repr__(self):
        return f"Workflow(id={self.id}, tasks={self.tasks})"

    def start(self):
        # Start the workflow and record start time
        self.start_time = time.time()

    def end(self):
        # End the workflow and record end time
        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time

# Define HeterogeneousCloudSimulator class
class HeterogeneousCloudSimulator:
    def __init__(self):
        self.workflows = []
        self.instance_types = []

    def add_instance_type(self, instance_type):
        self.instance_types.append(instance_type)

    def run_simulation(self):
        total_runtime = 0
        total_cost = 0
        for workflow in self.workflows:
            workflow.start()
            print(f"Starting Workflow {workflow.id}")

            # Assume we have a default instance type for simplicity
            default_instance = self.instance_types[0] if self.instance_types else None

            for task in workflow.tasks:
                if default_instance:
                    task.vm = default_instance
                    task.start(default_instance)
                    task.end()
                    print(f"Task {task.id} in Workflow {workflow.id} completed in {task.runtime:.2f} seconds with cost ${task.cost:.2f}")

                # Add up costs for this task
                total_cost += task.cost

            workflow.end()
            print(f"Workflow {workflow.id} completed in {workflow.total_runtime:.2f} seconds")
            total_runtime += workflow.total_runtime

        print(f"Total runtime for all workflows: {total_runtime:.2f} seconds")
        print(f"Total cost for all tasks: ${total_cost:.2f}")

    def visualize_workflows(self):
        output_dir = 'visualizations'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' created or already exists.")

        if not self.workflows:
            print("No workflows to visualize.")
            return

        for workflow in self.workflows:
            file_path = os.path.join(output_dir, f'workflow_{workflow.id}_dependencies.png')
            self.plot_task_dependencies(workflow, file_path)

    def plot_task_dependencies(self, workflow, file_path):
        G = nx.DiGraph()
        
        # Add nodes with labels including priority
        for task in workflow.tasks:
            label = f"Task {task.id}\nPriority {task.priority}".replace(":", "_").replace("\n", "\\n")
            G.add_node(task.id, label=label)

        # Add edges representing dependencies
        for task in workflow.tasks:
            for dep in workflow.get_dependencies(task):
                G.add_edge(dep.id, task.id)

        # Use spring layout as an alternative to graphviz layout
        pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility
        
        # Extract labels for nodes
        labels = nx.get_node_attributes(G, 'label')

        # Draw the graph with edges (lines between tasks)
        # nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        
        # # Set the title and save the figure
        # plt.title(f'Workflow {workflow.id} Dependencies')
        # plt.savefig(file_path)
        # plt.close()
        # print(f"Visualization saved to {file_path}")

def parse_xml_and_create_workflows(file_path, simulator):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespaces
    namespaces = {
        '': 'http://pegasus.isi.edu/schema/DAX'
    }

    # Find all job elements
    jobs = root.findall('job', namespaces)

    # Initialize a dictionary to store workflow mappings
    workflows = {}

    for job in jobs:
        # Extract job details
        job_id = job.get('id')
        name = job.get('name')
        version = job.get('version')
        runtime = float(job.get('runtime', 0))

        # Check if workflow already exists for this job
        if job_id not in workflows:
            workflow = Workflow(job_id)
            workflows[job_id] = workflow
        else:
            workflow = workflows[job_id]

        # Create and add tasks to the workflow
        for i, uses in enumerate(job.findall('uses', namespaces)):
            file_name = uses.get('file')
            link = uses.get('link')
            size = uses.get('size')

            # Example: Assign random resource requirements and length
            cpu_req = random.uniform(0.1, 8.0)
            gpu_req = random.choice([0, 1])
            fpga_req = random.choice([0, 1])
            length = int(runtime)  # Use runtime as length for simplicity
            priority = random.randint(1, 10)  # Random priority for each task

            task = Task(id=i+1,  # Unique ID for each task
                        resource_requirements={
                            ResourceType.CPU: cpu_req,
                            ResourceType.GPU: gpu_req,
                            ResourceType.FPGA: fpga_req
                        },
                        length=length,
                        priority=priority)
            workflow.add_task(task)

        # Add dependencies if needed
        # Example: You can implement dependencies based on your needs
        # For now, we'll assume no dependencies are added

    # Add workflows to the simulator
    for workflow_id, workflow in workflows.items():
        simulator.workflows.append(workflow)

# Example usage
def main():
    file_path = r'C:\Users\RGS\Desktop\resources\input\inputDAGfiles\inspiral300.xml'
    
    # Initialize simulator
    simulator = HeterogeneousCloudSimulator()

    # Add some instance types
    instance1 = InstanceType('Type1', cpu=4, gpu=1, fpga=0, cost_per_hour=0.5)
    instance2 = InstanceType('Type2', cpu=2, gpu=2, fpga=1, cost_per_hour=0.8)
    simulator.add_instance_type(instance1)
    simulator.add_instance_type(instance2)

    # Parse XML and create workflows
    parse_xml_and_create_workflows(file_path, simulator)

    # Run simulation and visualize
    simulator.run_simulation()
    simulator.visualize_workflows()

if __name__ == "__main__":
    main()
