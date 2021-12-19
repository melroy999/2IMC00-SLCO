import jinja2

# The jinja2 enviroment in which the rendering is done.
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("jinja2_templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"]
)
