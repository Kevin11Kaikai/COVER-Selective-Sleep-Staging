"""One contained entry point. Default replay never imports fitting code."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="2"
os.environ["MPLCONFIGDIR"]=str(ROOT/"outputs/cache/matplotlib")
os.environ["PYTHONPYCACHEPREFIX"]=str(ROOT/"outputs/cache/pycache")
os.environ["TMP"]=os.environ["TEMP"]=str(ROOT/"outputs/cache/tmp")
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"src"),str(ROOT.parent/"AutoML_Claude")]

def main():
    import argparse
    from new_objective.storage import Budget,OUT,write,utc
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("mode",nargs="?",default="replay",choices=["preflight","pilot","fit","evaluate","replay","report"])
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    (OUT/"cache/tmp").mkdir(parents=True,exist_ok=True)
    budget=Budget(args.mode)
    try:
        if args.mode in ("preflight","pilot","fit","evaluate"):
            from threadpoolctl import threadpool_limits
            from new_objective import experiment
            with threadpool_limits(limits=2):
                getattr(experiment,args.mode)(budget,args.resume)
        else:
            from new_objective import reporting
            getattr(reporting,args.mode)(budget)
        budget.finish("COMPLETED")
    except BaseException as error:
        budget.finish(type(error).__name__)
        write(OUT/"last_failure.json",{"time":utc(),"mode":args.mode,"type":type(error).__name__,"message":str(error)})
        raise

if __name__=="__main__":
    main()
