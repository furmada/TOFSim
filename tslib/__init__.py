import datetime
import os
from importlib import import_module

import_module

def ts_import(libname, version):
    available = list(sorted([l for l in os.listdir("tslib") if l.endswith(".py") and l.startswith(libname)]))
    if version == "latest":
        print("tslib." + available[-1].rstrip(".py"))
        return import_module("tslib." + available[-1].rstrip(".py"))
    for ver in available:
        if ver.rstrip(".py").lstrip(libname + "_") == version:
            return import_module("tslib." + ver.rstrip(".py"))
    raise ValueError("No such version: tslib/" + libname + "_" + version + ".py")

def ts_output(progid):
    dt = datetime.datetime.now()
    path = os.path.join("results", progid + dt.strftime("%Y%m%d"))
    if os.path.exists(path):
        print("Results directory already exists:", path)
        with open(os.path.join(path, "log.txt"), "a") as f:
            f.write("\nOpened output folder from " + progid + " at " + dt.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print("Creating results directory:", path)
        os.mkdir(path)
        with open(os.path.join(path, "log.txt"), "w") as f:
            f.write("Created output folder from " + progid + " at " + dt.strftime("%Y-%m-%d %H:%M:%S"))
    return path

def ts_log(progid, *messages):
    dt = datetime.datetime.now()
    path = os.path.join("results", progid + dt.strftime("%Y%m%d"))
    if os.path.exists(path):
        with open(os.path.join(path, "log.txt"), "a") as f:
            f.write("\n" + dt.strftime("%Y-%m-%d %H:%M:%S") + ": " + " ".join([str(m) for m in messages]))
            print(*messages)
    else:
        raise ValueError("The output folder was not made yet!")

def ts_csvout(progid, fname, data=[]):
    dt = datetime.datetime.now()
    path = os.path.join("results", progid + dt.strftime("%Y%m%d"))
    if os.path.exists(path):
        with open(os.path.join(path, fname), "a") as f:
            f.write(",".join([str(d) for d in data]) + "\n")
    else:
        raise ValueError("The output folder was not made yet!")