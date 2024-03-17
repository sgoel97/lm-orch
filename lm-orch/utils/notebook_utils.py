def running_in_notebook():
    try:
        from IPython import get_ipython

        # Check if get_ipython exists and if it's inside a notebook
        if "IPKernelApp" not in get_ipython().config:  # Not in a notebook
            return False
    except (ImportError, AttributeError):
        return False  # IPython is not installed
    return True  # IPython exists, assume it's a notebook
