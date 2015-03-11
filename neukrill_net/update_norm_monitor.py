"""
Defines a pylearn2 learning rule wrapper class for monitoring of the norms
of gradient based updates.
"""

import six
from pylearn2.training_algorithms.learning_rule import LearningRule
from pylearn2.utils import sharedX
import theano.tensor as T

class UpdateNormMonitorLearningRule(LearningRule):

    """ Wraps an existing pylearn2 learning rule and adds monitor channels
        for the norms of the gradient based updates calculated during
        learning.
    """
    
    def __init__(self, base_learning_rule, decay=0.9):
         self.base = base_learning_rule
         self.decay = decay
         self.mean_updates = OrderedDict()
         
    def add_channels_to_monitor(self, monitor, monitoring_dataset):
    
        channel_mapping = {
            '_min': T.min,
            '_max': T.max,
            '_mean': T.mean
        }
        
        for mean_update in self.grad_norms.values():
            if mean_update.ndim == 4:
                # rank-4 tensor (assuming stack of rank-3 convolutional kernels)
                knl_norm_vals = T.sqrt(T.sum(T.sqr(mean_update), axis=(1,2,3)))
                for suffix, op in channel_mapping:
                    monitor.add_channel(
                        name=(mean_update.name + "_kernel_norm" + suffix),
                        ipt=None,
                        val=op(knl_norm_vals)
                        data_specs=(NullSpace(), ''),
                        dataset=monitoring_dataset)
            elif mean_update.ndim == 2:
                # rank-2 tensor (matrix)
                col_norm_vals = T.sqrt(T.sum(T.sqr(mean_update), axis=0))
                row_norm_vals = T.sqrt(T.sum(T.sqr(mean_update), axis=1))
                mtx_norm_val = T.sqrt(T.sum(T.sqr(mean_update))        
                for suffix, op in channel_mapping:
                    monitor.add_channel(
                        name=(mean_update.name + "_col_norm" + suffix),
                        ipt=None,
                        val=op(col_norm_vals)
                        data_specs=(NullSpace(), ''),
                        dataset=monitoring_dataset)
                    monitor.add_channel(
                        name=(mean_update.name + "_row_norm" + suffix),
                        ipt=None,
                        val=op(row_norm_vals)
                        data_specs=(NullSpace(), ''),
                        dataset=monitoring_dataset)
                monitor.add_channel(
                    name=(mean_update.name + "_norm"),
                    ipt=None,
                    val=mtx_norm_val,
                    data_specs=(NullSpace(), ''),
                    dataset=monitoring_dataset)
            elif mean_update.ndim == 1:
                # rank-1 tensor (vector)
                norm_val = T.sqrt(T.sum(T.sqr(mean_update), axis=0))
                monitor.add_channel(
                    name=(mean_update.name + "_norm"),
                    ipt=None,
                    val=norm_val,
                    data_specs=(NullSpace(), ''),
                    dataset=monitoring_dataset)
            elif mean_update.ndim == 0:
                # rank-0 tensor (scalar)
                monitor.add_channel(
                    name=(mean_update.name + "_norm"),
                    ipt=None,
                    val=mean_update,
                    data_specs=(NullSpace(), ''),
                    dataset=monitoring_dataset)                
            else:
                # not sure which axes to sum over in this case
                raise ValueError(
                    'Mean update {0} has unexpected number of dimensions {0}'
                    .format(mean_update, mean_update.ndim))
                    
        self.base.add_channels_to_monitor(monitor, monitoring_dataset)
        
        return  

    def get_updates(self, learning_rate, grads, lr_scalers=None):
    
        updates = self.base.get_updates(learning_rate, grads, lr_scalers)
    
        for (param, grad) in six.iteritems(grads):

            mean_update = sharedX(param.get_value() * 0.)

            if param.name is None:
                raise ValueError("Model parameters must be named.")
            mean_update.name = 'mean_update_' + param.name

            if param.name in self.mean_updates:
                warnings.warn("Calling get_updates more than once on the "
                              "gradients of `%s` may make monitored values "
                              "incorrect." % param.name)
                              
            # Store variable in self.mean_updates for monitoring.
            self.mean_updates[param.name] = mean_updates

            # Accumulate updates
            d_param = updates[param] - param
            new_mean_update = (self.decay * mean_update + 
                               (1 - self.decay) * d_param)

            # Apply update
            updates[mean_update] = new_mean_update
            
        return updates
            
            
            
        
        
