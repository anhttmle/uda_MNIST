import time


def get_time_process(target_function):
    start_time = time.time()
    target_function()
    end_time = time.time()

    return end_time - start_time


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) contains {} nodes in its graph".format(
        f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)
    ))
    return f

