# es_framework/optimizers/__init__.py

from .cem import CEMOptimizer
# from .cma_es import CMAESOptimizer  # Uncomment when CMA is ready


def get_optimizer(optimizer_name, config, policy_structure):
    """
    Factory function to initialize the requested optimizer.
    
    Args:
        optimizer_name (str): 'CEM' or 'CMA'
        config (dict): Configuration dictionary containing hyperparameters
        policy_structure (Policy): The neural network model (to get param dims)
        
    Returns:
        An instance of the requested optimizer.
    """
    name = optimizer_name.upper()

    if name == 'CEM':
        return CEMOptimizer(config, policy_structure)

    elif name == 'CMA':
        # return CMAESOptimizer(config, policy_structure)
        raise NotImplementedError("CMA-ES is not fully linked yet. Uncomment import first.")

    else:
        raise ValueError(f"Unknown Optimizer: {name}. Available: ['CEM', 'CMA']")
