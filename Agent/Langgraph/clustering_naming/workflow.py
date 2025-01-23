class ClusterNamingWorkflow:
    def __init__(self):
        self.state = {
            "clusters": [],
            "cluster_names": [],
            "current_step": 0
        }

    def add_cluster(self, cluster):
        self.state["clusters"].append(cluster)

    def set_cluster_name(self, cluster_id, name):
        if cluster_id < len(self.state["clusters"]):
            self.state["cluster_names"].append(name)

    def next_step(self):
        self.state["current_step"] += 1

    def get_state(self):
        return self.state
