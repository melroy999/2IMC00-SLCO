# Import the SLCO2SLCO module.
import importlib
slco2slco = importlib.import_module("SLCOtoSLCO.python-textx-jinja2.slco2slco")


# Hotfixes.
def RepresentsInt(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


# Overwrite functions with hotfixed versions.
slco2slco.RepresentsInt = RepresentsInt


def simplify_slco_model(model):
    """Use the SLCO2SLCO module to convert the slco model to a simplified model."""
    slco2slco.model = model
    slco2slco.preprocess()

    # Apply the desired transformations.
    model = slco2slco.model
    # slco2slco.check_vars(model)
    slco2slco.check_repeatedwrites(model)
    # slco2slco.combine_trans(model) TODO: Causes statements to disappear?
    slco2slco.make_simple(model)

    return model
