# The current id.
current_id = 0


def get_incremental_id() -> int:
    """
    Get an incremental id.
    """
    global current_id
    try:
        return current_id
    finally:
        current_id += 1
