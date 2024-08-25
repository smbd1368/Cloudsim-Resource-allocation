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
        self.length = length
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
        self.workflow_id_counter = 0
        self.instance_types = []  # List of available instance types

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

    def add_instance_type(self, instance_type):
        self.instance_types.append(instance_type)

    def run_simulation(self):
        print("Running simulation...")
        # Call genetic algorithm to schedule tasks
        self.genetic_algorithm_scheduling()

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
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        
        # Set the title and save the figure
        plt.title(f'Workflow {workflow.id} Dependencies')
        plt.savefig(file_path)
        plt.close()
        print(f"Visualization saved to {file_path}")

    def genetic_algorithm_scheduling(self):
        print("Running Genetic Algorithm for scheduling...")

        def initialize_population(size):
            population = []
            for _ in range(size):
                # Generate random allocation of tasks to instance types
                schedule = [random.choice(self.instance_types) for _ in range(len(self.workflows[0].tasks))]
                population.append(schedule)
            return population

        def fitness(schedule):
            total_cost = 0
            for task, instance_type in zip(self.workflows[0].tasks, schedule):  # Simplified: assumes one workflow
                task.vm = instance_type  # Assign the VM
                task.start(instance_type)
                total_cost += task.cost
                task.end()
            return total_cost

        def select(population, fitnesses):
            # Pair each individual with its fitness and sort based on fitness values
            paired = sorted(zip(fitnesses, population), key=lambda x: x[0])
            return [x for _, x in paired[:len(population) // 2]]

        def crossover(parent1, parent2):
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2

        def mutate(schedule):
            idx = random.randint(0, len(schedule) - 1)
            schedule[idx] = random.choice(self.instance_types)
            return schedule

        # GA parameters
        population_size = 10
        generations = 10

        # Initialize population
        population = initialize_population(population_size)

        best_fitness_per_generation = []

        for generation in range(generations):
            print(f"Generation {generation + 1}")

            # Evaluate fitness
            fitnesses = [fitness(schedule) for schedule in population]
            print(f"Fitnesses: {fitnesses}")

            # Track the best fitness in this generation
            best_fitness = min(fitnesses)
            best_fitness_per_generation.append(best_fitness)
            print(f"Best fitness in generation {generation + 1}: {best_fitness}")

            # Select the best individuals
            selected = select(population, fitnesses)

            # Create new population through crossover and mutation
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1))
                new_population.append(mutate(child2))
            
            # Ensure population size
            population = selected + new_population[:population_size - len(selected)]

        # Evaluate final population
        fitnesses = [fitness(schedule) for schedule in population]
        best_schedule = population[fitnesses.index(min(fitnesses))]
        best_fitness = min(fitnesses)
        print(f"Best schedule fitness: {best_fitness}")

        # Plot convergence
        plt.plot(range(1, generations + 1), best_fitness_per_generation, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Value')
        plt.title('Convergence of Genetic Algorithm')
        plt.grid(True)
        plt.savefig('convergence_plot.png')
        plt.show()
# Example usage
if __name__ == "__main__":
    simulator = HeterogeneousCloudSimulator()

    # Create some instance types
    instance1 = InstanceType('Type1', cpu=4, gpu=1, fpga=0, cost_per_hour=0.5)
    instance2 = InstanceType('Type2', cpu=2, gpu=2, fpga=1, cost_per_hour=0.8)
    simulator.add_instance_type(instance1)
    simulator.add_instance_type(instance2)

    # Create and visualize workflows 1 to 10
    for i in range(10):
        workflow = simulator.create_and_add_workflow()
        previous_task = None
        for j in range(10):  # Example number of tasks per workflow
            cpu_req = random.uniform(0.1, 8.0)
            gpu_req = random.choice([0, 1])
            fpga_req = random.choice([0, 1])
            length = random.randint(1000, 5000)  # Increased length to simulate more realistic runtimes
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
