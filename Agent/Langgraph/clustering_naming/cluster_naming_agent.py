from langgraph.orchestration import Orchestrator

class ClusterNamingAgent:
    def __init__(self):
        self.cluster_names = []
        self.orchestrator = Orchestrator()

    def generate_cluster_name(self, cluster_id):
        # Placeholder for cluster naming logic
        return f"Cluster_{cluster_id}"

    def name_clusters(self, num_clusters):
        for i in range(num_clusters):
            cluster_name = self.generate_cluster_name(i)
            self.cluster_names.append(cluster_name)
        return self.cluster_names

    def execute(self, num_clusters):
        # Use the orchestrator to handle the execution
        return self.orchestrator.run(self.name_clusters, num_clusters)
