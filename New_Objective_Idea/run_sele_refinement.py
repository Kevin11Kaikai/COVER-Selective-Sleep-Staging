"""Bounded SELE-reference FIT diagnostic. No CERT/OUTER evaluation mode exists."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/cache/matplotlib')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'outputs/cache/tmp')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'src'), str(ROOT.parent / 'AutoML_Claude')]


def main():
    import argparse
    from threadpoolctl import threadpool_limits
    from new_objective import sele_refinement as experiment
    from new_objective.storage import Budget, write, utc
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', nargs='?', default='replay',
                        choices=['preflight', 'pilot', 'fit', 'replay'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    budget = Budget(args.mode)
    budget.state['sessions'][budget.index]['experiment'] = 'sele-refinement-fit-v1'
    try:
        with threadpool_limits(limits=2):
            getattr(experiment, args.mode)(budget, args.resume)
        budget.finish('COMPLETED')
    except BaseException as error:
        budget.finish(type(error).__name__)
        write(experiment.SOUT / 'last_failure.json',
              {'time': utc(), 'mode': args.mode, 'error': type(error).__name__,
               'message': str(error)})
        raise


if __name__ == '__main__':
    main()
