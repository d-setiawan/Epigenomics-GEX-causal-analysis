import os

from filelock import FileLock
from joblib import Parallel, delayed

from .estimator import DAGEstimator


class ParallelDAGEstimator(DAGEstimator):
    def run_and_log(self, i, truth_graph, output_path):
        try:
            res = self.estimate(i, truth_graph)
        except Exception as e:
            print(f"[Warning] Iteration {i} failed: {e}")
            res = ("ERROR", "ERROR", "ERROR")

        res_line = [str(i)] + [f"{x:.6f}" if isinstance(x, float) else str(x) for x in res]
        line = ",".join(res_line) + "\n"

        with FileLock(output_path + ".lock"):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(line)

    def parallel_estimate_streaming(self, num_iterations, truth_graph, output_path=None, n_jobs=5):
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        if output_path is None:
            filename = f"results_{self.N}_{self.distribution}_{self.DAG_type}_{self.Q_type}.txt"
            output_path = os.path.join(results_dir, filename)

        if not os.path.exists(output_path):
            with open(output_path, "w", encoding="utf-8") as f:
                header = ["iter", "shd_val", "shd2_val", "shd3_val"]
                f.write(",".join(header) + "\n")

        Parallel(n_jobs=n_jobs)(
            delayed(self.run_and_log)(i, truth_graph, output_path)
            for i in num_iterations
        )
