
def get_and_remove_up_to_first(string, sub, skip):
	idx = string.find(sub)
	if idx == -1:
		return string, ""

	ret1 = string[:idx]
	ret2 = string[idx + skip:]
	return ret1, ret2


def get_read_vars_from_expression(expr, acc):
	if not expr:
		return
	expr1 = expr.left.left.left.left
	if expr1:
		primary = expr1.left
		if primary and primary.ref:
			acc.add(primary.ref.ref)
	get_read_vars_from_expression(expr.right, acc)