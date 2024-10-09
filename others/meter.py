import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CostMeter(object):
    def __init__(self, init_cost=None):
        self.reset()
        if init_cost is not None:
            self.best_cost = init_cost
            self.best_step = 0

    def reset(self):
        self.best_cost = np.inf
        self.best_step = None
        self.actual_acc_cost, self.planned_acc_cost = 0, 0
        self.actual_cost_seq, self.planned_cost_seq, self.step_seq = [], [], []

    def update(self, actual_cost, planned_cost, step):
        self.actual_acc_cost += actual_cost
        self.planned_acc_cost += planned_cost
        self.actual_cost_seq.append(actual_cost)
        self.planned_cost_seq.append(planned_cost)
        self.step_seq.append(step)
        if actual_cost < self.best_cost:
            self.best_cost = actual_cost
            self.best_step = step
        print(
            f"Step {step}: actual cost {actual_cost}, planned cost {planned_cost}, cost diff {actual_cost - planned_cost}"
        )
        return

    def print_metric(self):
        actual_acc_cost = self.actual_acc_cost
        planned_acc_cost = self.planned_acc_cost
        print(
            f"Actual accumulated cost: {actual_acc_cost}, planned accumulated cost: {planned_acc_cost}, accumulated cost diff: {actual_acc_cost - planned_acc_cost}"
        )
        print(f"final cost: {self.actual_cost_seq[-1]}, best cost: {self.best_cost}, best step: {self.best_step}")

    def get_metric(self):
        return {
            "best_cost": self.best_cost,
            "best_step": self.best_step,
            "actual_accumulated_cost": self.actual_acc_cost,
            "planned_accumulated_cost": self.planned_acc_cost,
            "accumulated_cost_diff": self.actual_acc_cost - self.planned_acc_cost,
            "actual_cost_seq": self.actual_cost_seq,
            "planned_cost_seq": self.planned_cost_seq,
            "cost_diff_seq": np.subtract(self.actual_cost_seq, self.planned_cost_seq).tolist(),
        }
