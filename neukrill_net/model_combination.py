"""
Creates a theano based gradient descent optimiser for finding good choices of
weights to combine model predictions.
"""

import theano as th
import theano.tensor as tt
import numpy as np

def compile_model_combination_weight_optimiser(lr_adjuster = lambda h, t: h):
    model_weights = tt.vector('w') # indexed over K models
    model_preds = tt.tensor3('P') # indexed over N examples, M classes, K models
    true_labels = tt.matrix('Y') # indexed over N examples, M classes
    learning_rate = tt.scalar('h')
    n_steps = tt.iscalar('n_steps')
    # use softmax form to ensure weights all >=0 and sum to one
    comb_preds = (tt.sum(tt.exp(model_weights) * model_preds, axis=2) / 
                  tt.sum(tt.exp(model_weights), axis=0))
    # mean negative log loss cost function
    cost = - tt.mean(tt.sum(tt.log(comb_preds) * true_labels, axis=1))
    # gradient of log loss cost with respect to weights
    dC_dW = lambda W: th.clone(th.gradient.jacobian(cost, model_weights), 
                               {model_weights: W})
    # scan through gradient descent updates of weights, applying learning rate
    # adjuster at each step
    [Ws, hs], updates = th.scan(
         fn = lambda t, W, h: [W - h * dC_dW(W), lr_adjuster(h, t)],
         outputs_info = [model_weights, learning_rate],
         sequences = [th.tensor.arange(n_steps)],
         n_steps = n_steps,
         name = 'weight cost gradient descent')
    # create a function to get last updated weight from scan sequence
    weights_optimiser = th.function(
        inputs = [model_weights, model_preds, true_labels, learning_rate, 
                  n_steps],
        outputs = Ws[-1],
        updates = updates,
        allow_input_downcast = True,
    )
    # also compile a function for evaluating cost function to check optimiser
    # performance / convergence
    cost_func = th.function([model_weights, model_preds, true_labels], cost)
    return weights_optimiser, cost_func
    
if __name__ == '__main__':

    """
    Test with randomly generated model predictions and labels.
    """

    N_MODELS = 3
    N_CLASSES = 10
    N_DATA = 100
    SEED = 1234
    INIT_LEARNING_RATE = 0.1
    LR_ADJUSTER = lambda h, t: h
    N_STEP = 1000

    prng = np.random.RandomState(SEED)
    weights = np.zeros(3)
    model_pred_vals = prng.rand(N_DATA, N_CLASSES, N_MODELS)
    model_pred_vals = model_pred_vals /  model_pred_vals.sum(1)[:,None,:]
    true_label_vals = prng.rand(N_DATA, N_CLASSES)
    true_label_vals = true_label_vals == true_label_vals.max(axis=1)[:,None]
    
    optimiser, cost = compile_model_combination_weight_optimiser(LR_ADJUSTER)

    print('Initial weights {0}'.format(weights))
    print('Initial cost value {0}'.format(
        cost(weights, model_pred_vals, true_label_vals)))
    
    updated_weights = optimiser(weights, model_pred_vals, true_label_vals, 
                                INIT_LEARNING_RATE, N_STEP)
    
    print('Final weights {0}'.format(updated_weights))
    print('Final cost value {0}'.format(
        cost(updated_weights, model_pred_vals, true_label_vals)))

