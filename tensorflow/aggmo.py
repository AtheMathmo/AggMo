"""AggMo for TensorFlow."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer


class AggMo(optimizer.Optimizer):
    def __init__(self, learning_rate=0.1, betas=[0, 0.9, 0.99], use_locking=False, name="AggMo"):
        super(AggMo, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._betas = betas
        
    @classmethod
    def from_exp_form(cls, learning_rate, a=0.1, k=3, use_locking=False, name="AggMo"):
        betas = [1.0 - a**i for i in range(K)]
        return cls(learning_rate, betas, use_locking, name)

    def _create_slots(self, var_list):
        # Create slots for each momentum component
        for v in var_list :
            for i in range(len(self._betas)):
                self._zeros_slot(v, "momentum_{}".format(i), self._name)

    def _prepare(self):
        learning_rate = self._lr
        if callable(learning_rate):
            learning_rate = learning_rate()
        self._lr_tensor = ops.convert_to_tensor(learning_rate, name="learning_rate")

        betas = self._betas
        if callable(betas):
            betas = betas()
        self._betas_tensor = ops.convert_to_tensor(betas, name="betas")

    def _apply_dense(self, grad, var):
        lr = math_ops.cast(self._lr_tensor / len(self._betas), var.dtype.base_dtype)
        betas = math_ops.cast(self._betas_tensor, var.dtype.base_dtype)

        momentums = []
        summed_momentum = 0.0
        for i in range(len(self._betas)):
            m = self.get_slot(var, "momentum_{}".format(i))
            m_t = state_ops.assign(m, betas[i] * m + grad)
            summed_momentum += m_t
            momentums.append(m_t)
        var_update = state_ops.assign_sub(var, lr * summed_momentum, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, *momentums])

    def _resource_apply_dense(self, grad, var):
        var = var.handle
        lr = math_ops.cast(self._lr_tensor / len(self._betas), var.dtype.base_dtype)
        betas = math_ops.cast(self._betas_tensor, var.dtype.base_dtype)

        momentums = []
        summed_momentum = 0.0
        for i in range(len(self._betas)):
            m = self.get_slot(var, "momentum_{}".format(i))
            m_t = state_ops.assign(m, betas[i] * m + grad)
            summed_momentum += m_t
            momentums.append(m_t)
        var_update = state_ops.assign_sub(var, lr * summed_momentum, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, *momentums])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        lr = math_ops.cast(self._lr_tensor / len(self._betas), var.dtype.base_dtype)
        betas = math_ops.cast(self._betas_tensor, var.dtype.base_dtype)

        momentums = []
        summed_momentum = 0.0
        for i in range(len(self._betas)):
            m = self.get_slot(var, "momentum_{}".format(i))
            m_t = state_ops.assign(m, betas[i] * m, use_locking=self._use_locking)
            with ops.control_dependencies([m_t]):
                m_t = scatter_add(m, indices, grad)
            momentums.append(m_t)
            summed_momentum += m_t
        var_update = state_ops.assign_sub(var, lr * summed_momentum, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, *momentums])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)
