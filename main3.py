import random
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from enum import Enum
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
from networkx.drawing.nx_pydot import pydot_layout


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
        self.length = length
        self.dependencies = set()
        self.vm = None  # To track which VM executed this task
        self.priority = priority  # New attribute for priority

    def add_dependency(self, task):
        self.dependencies.add(task)

    def __repr__(self):
        return f"Task(id={self.id}, priority={self.priority})"

# Define Workflow class
class Workflow:
    def __init__(self, id):
        self.id = id
        self.tasks = []
        self.task_dict = {}  # To quickly lookup tasks by their id

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

# Define HeterogeneousCloudSimulator class
class HeterogeneousCloudSimulator:
    def __init__(self):
        self.workflows = []
        self.workflow_id_counter = 0

    def create_and_add_workflow(self):
        self.workflow_id_counter += 1
        new_workflow = Workflow(self.workflow_id_counter)
        self.workflows.append(new_workflow)
        print(f"Created and added Workflow {new_workflow.id}")  # Debugging line
        return new_workflow

    def create_and_add_task(self, workflow, cpu_req, gpu_req, fpga_req, length, priority=0):
        task_id = len(workflow.tasks) + 1
        resource_requirements = {
            ResourceType.CPU: cpu_req,
            ResourceType.GPU: gpu_req,
            ResourceType.FPGA: fpga_req
        }
        task = Task(task_id, resource_requirements, length, priority)
        workflow.add_task(task)
        print(f"Added Task {task.id} to Workflow {workflow.id} with priority {priority}")  # Debugging line
        return task

    def run_simulation(self):
        print("Running simulation...")
        # Simulation logic here
        print("Simulation completed.")

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
            # Replace any problematic characters with underscores
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
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        
        # Set the title and save the figure
        plt.title(f'Workflow {workflow.id} Dependencies')
        plt.savefig(file_path)
        plt.close()
        print(f"Visualization saved to {file_path}")
# Example usage# Example usage
if __name__ == "__main__":
    simulator = HeterogeneousCloudSimulator()

    # Create and visualize workflows 1 to 10
    for i in range(10):
        workflow = simulator.create_and_add_workflow()
        previous_task = None
        for j in range(10):  # Example number of tasks per workflow
            cpu_req = random.uniform(0.1, 8.0)
            gpu_req = random.choice([0, 1])
            fpga_req = random.choice([0, 1])
            length = random.randint(5000, 20000)
            priority = random.randint(1, 10)  # Random priority for each task
            task = simulator.create_and_add_task(workflow, cpu_req, gpu_req, fpga_req, length, priority)
            
            # Add dependencies
            if previous_task:
                workflow.add_dependency(task.id, previous_task.id)
            if j % 3 == 0:  # Create some branching
                previous_task = task
            if j % 5 == 0:  # Reset occasionally
                previous_task = None

    # Run simulation and visualize
    simulator.run_simulation()
    simulator.visualize_workflows()
