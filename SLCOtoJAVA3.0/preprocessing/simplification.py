# Import the SLCO2SLCO module.
import importlib
slco2slco = importlib.import_module("SLCOtoSLCO.python-textx-jinja2.slco2slco")


# Hotfixes.
def represents_int(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


# Overwrite functions with corrected versions.
slco2slco.RepresentsInt = represents_int


# noinspection PyUnresolvedReferences
def simplify_slco_model(model):
    """Use the SLCO2SLCO module to convert the slco model to a simplified model."""
    slco2slco.model = model
    slco2slco.preprocess()

    # Apply the desired transformations.
    model = slco2slco.model
    # slco2slco.check_vars(model)
    # TODO: Removed due to itmp'0 occurring multiple times when repeating i := i + 1; i := i + 1 in composite.
    # slco2slco.check_repeatedwrites(model)
    # slco2slco.combine_trans(model) TODO: Causes statements to disappear?
    slco2slco.make_simple(model)

    return model
