import copy
import numpy as np
from .base import BaseOptimization

class DOO(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)
        
    def single_rollout(self, X, x_current, rollout_round, top_n=16, top_n2=4, method_args=dict(explr_p=0.01)):
        domain = np.array([[self.f.lb[0]] * self.f.dims, [self.f.ub[0]] * self.f.dims])
        distance_fn = lambda x, y: np.linalg.norm(x - y)
        doo_tree = BinaryDOOTree(domain=domain, distance_fn=distance_fn, **method_args)

        for i in range(rollout_round):
            next_node = doo_tree.get_next_point_and_node_to_evaluate()
            x_to_evaluate = next_node.cell_mid_point
            next_node.evaluated_x = x_to_evaluate
            x_to_evaluate=np.round(x_to_evaluate,int(-np.log10(self.f.turn)))
            fval = self.model.predict(np.array(x_to_evaluate).reshape(1,self.f.dims,1))
            fval = np.array(fval).reshape(1)
            doo_tree.expand_node(fval, next_node)
            self.all_proposed.append(x_to_evaluate)

        return self.get_top_X(X, top_n, top_n2)
    

class DOOTreeNode:
    def __init__(self, cell_mid_point, cell_min, cell_max, parent_node, distance_fn, idx):
        self.cell_mid_point = cell_mid_point
        self.evaluated_x = None
        self.l_child = None
        self.r_child = None
        self.cell_min = cell_min  # size of the cell
        self.cell_max = cell_max
        self.delta_h = distance_fn(cell_mid_point, self.cell_min) + distance_fn(cell_mid_point, self.cell_min)
        self.parent = parent_node
        self.f_value = None
        self.idx = idx

    def update_node_f_value(self, fval):
        self.f_value = fval


class BinaryDOOTree:
    def __init__(self, domain, explr_p, distance_fn):
        self.root = None
        self.leaves = []
        self.nodes = []
        self.domain = domain
        self.distance_fn = distance_fn
        self.explr_p = explr_p
        self.node_to_update = None
        self.evaled_x_to_node = {}

    def create_node(self, cell_mid_point, cell_min, cell_max, parent_node):
        new_node = DOOTreeNode(cell_mid_point, cell_min, cell_max, parent_node, self.distance_fn, idx=len(self.nodes))
        return new_node

    def add_left_child(self, parent_node):
        left_child_cell_mid_point_x = self.compute_left_child_cell_mid_point(parent_node)
        cell_min, cell_max = self.compute_left_child_cell_limits(parent_node)

        node = self.create_node(left_child_cell_mid_point_x, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'left')

    def add_right_child(self, parent_node):
        right_child_cell_mid_point_x = self.compute_right_child_cell_mid_point(parent_node)
        cell_min, cell_max = self.compute_right_child_cell_limits(parent_node)

        node = self.create_node(right_child_cell_mid_point_x, cell_min, cell_max, parent_node)
        self.add_node_to_tree(node, parent_node, 'right')

    def find_leaf_with_max_upper_bound_value(self):
        max_upper_bound = -np.inf
        for leaf_node in self.leaves:
            if leaf_node.f_value is None:
                return leaf_node
            if leaf_node.f_value == 'update_me':
                continue
            node_upper_bound = leaf_node.f_value + self.explr_p*leaf_node.delta_h
            if node_upper_bound > max_upper_bound:
                best_leaf = leaf_node
                max_upper_bound = node_upper_bound
        is_node_children_added = not(best_leaf.l_child is None)
        if is_node_children_added:
            is_left_child_evaluated = best_leaf.l_child.f_value is not None
            is_right_child_evaluated = best_leaf.r_child.f_value is not None
            if not is_left_child_evaluated:
                return best_leaf.l_child
            elif not is_right_child_evaluated:
                return best_leaf.r_child
            else:
                assert False, 'When both children have been evaluated, the node should not be in the self.leaves'
        else:
            return best_leaf

    def get_next_point_and_node_to_evaluate(self):
        is_first_evaluation = self.root is None
        dim_domain = len(self.domain[0])
        if is_first_evaluation:
            cell_mid_point = np.random.uniform(self.domain[0], self.domain[1], (1, dim_domain)).squeeze()
            node = self.create_node(cell_mid_point, self.domain[0], self.domain[1], None)
            self.leaves.append(node)
            self.root = node
        else:
            node = self.find_leaf_with_max_upper_bound_value()
        return node

    def update_evaled_x_to_node(self, x, node):
        self.evaled_x_to_node[tuple(x)] = node

    def expand_node(self, fval, node):
        if fval == 'update_me':
            self.node_to_update = node
        else:
            self.node_to_update = None

        node.update_node_f_value(fval)
        self.nodes.append(node)

        self.add_left_child(node)
        self.add_right_child(node)

        if node.parent is not None:
            is_parent_node_children_all_evaluated = node.parent.l_child.f_value is not None \
                                                    and node.parent.r_child.f_value is not None
            if is_parent_node_children_all_evaluated:
                self.add_to_leaf(node.parent.l_child)
                self.add_to_leaf(node.parent.r_child)

    def add_to_leaf(self, node):
        parent_node = node.parent
        self.leaves.append(node)
        if parent_node in self.leaves:
            self.leaves.remove(parent_node)

    def find_evaled_f_value(self, target_x_value, evaled_x, evaled_y):
        # it all gets stuck here most of the time.
        # This is likely because there are so many self.nodes and that there are so many evaled_x
        # create a mapping between the node to the evaled_x value
        is_in_array = [np.all(np.isclose(target_x_value, a)) for a in evaled_x]
        is_action_included = np.any(is_in_array)
        assert is_action_included, 'action that needs to be updated does not have a value'
        return evaled_y[np.where(is_in_array)[0][0]]

    def update_evaled_values(self, evaled_x, evaled_y, infeasible_reward, idx_to_update):
        # Updates the evaluated x values in the tree
        if len(evaled_x) == 0:
            return

        feasible_idxs = np.zeros((len(evaled_x,)), dtype=bool)
        feasible_idxs[idx_to_update] = True
        feasible_idxs = np.array(feasible_idxs)

        evaled_x_to_update = np.array(evaled_x)[feasible_idxs, :]  # only the feasible ones get their f values updated
        evaled_y_to_update = np.array(evaled_y)[feasible_idxs]
        for x, y in zip(evaled_x_to_update, evaled_y_to_update):
            node_to_update = self.evaled_x_to_node[tuple(x)]
            node_to_update.f_value = y

        fvals_in_tree = np.array([n.f_value for n in self.nodes])
        sorted_evaled_y = np.array(evaled_y)
        assert np.array_equal(fvals_in_tree.sort(), sorted_evaled_y.sort()), "Are you using N_r?"

    @staticmethod
    def add_node_to_tree(node, parent_node, side):
        node.parent = parent_node
        if side == 'left':
            parent_node.l_child = node
        else:
            parent_node.r_child = node

    @staticmethod
    def compute_left_child_cell_mid_point(node):
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (node.cell_min[cutting_dimension] + node.cell_mid_point[cutting_dimension]) / 2.0
        return cell_mid_point

    @staticmethod
    def compute_right_child_cell_mid_point(node):
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (node.cell_max[cutting_dimension] + node.cell_mid_point[cutting_dimension]) / 2.0

        return cell_mid_point

    @staticmethod
    def compute_left_child_cell_limits(node):
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_min = copy.deepcopy(node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_max[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max

    @staticmethod
    def compute_right_child_cell_limits(node):
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_min = copy.deepcopy(node.cell_min)
        cell_min[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max