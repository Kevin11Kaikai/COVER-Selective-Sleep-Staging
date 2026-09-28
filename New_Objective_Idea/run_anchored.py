"""Contained anchored-risk extension. Default mode replays saved outcomes without fitting."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/cache/matplotlib')
os.environ['TMP']=os.environ['TEMP']=str(ROOT/'outputs/cache/tmp')
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'src'),str(ROOT.parent/'AutoML_Claude')]


def main():
    import argparse
    from new_objective.storage import Budget,write,OUT,utc
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',nargs='?',default='replay',choices=['preflight','pilot','fit','evaluate','replay','report'])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    budget=Budget(args.mode)
    budget.state['sessions'][budget.index]['experiment']='anchored-risk-v1'
    try:
        if args.mode in ('replay','report'):
            from new_objective import anchor_report as module
        else:
            from new_objective import anchor_experiment as module
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):getattr(module,args.mode)(budget,args.resume)
        budget.finish('COMPLETED')
    except BaseException as error:
        budget.finish(type(error).__name__)
        write(OUT/'anchored/last_failure.json',{'time':utc(),'mode':args.mode,'error':type(error).__name__,'message':str(error)})
        raise


if __name__=='__main__':main()
