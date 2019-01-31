#/* Gray Thomas, UT Austin---NSTRF NNX15AQ33H---Summer 2017 */
"""
pickle memoize introduces the @memoize decorator.

This decorator memoizes a function, using pkl to store outputs for re-use.
memoiziation comparison is based on the string passed in as hasharg.

hasharg must be a kwarg of the decorated function.

hard-coded PKL_FOLDER path configures the storage location.

"""
import pickle as pkl
import os.path


PKL_FOLDER = "/home/gray/wk/pkl/"
# PKL_FOLDER = "/home/gray/wk/pkl/"
def memoize(filename, override=False):
    """ Decorator to memoize a function. """
    def funcgen(fctn):
        """ generates a memoized replacement for fctn. """
        def newfctn(*args, hasharg=None, **kwargs):
            """ Function replacement. """
            fullname = filename+hasharg+".pkl"
            print("memoizing %s as"%fctn.__name__, fullname)
            if override or not os.path.isfile(PKL_FOLDER+fullname):
                res = fctn(*args, hasharg=hasharg, **kwargs)
                print("regen %s"%fullname)
                with open(PKL_FOLDER+fullname, 'wb') as fil:
                    pkl.dump(res, fil)
            else:
                print("load %s"%fullname)
                with open(PKL_FOLDER+fullname, 'rb') as fil:
                    res = pkl.load(fil)
            return res
        return newfctn
    return funcgen

def simple_memoize(filename, data_gen, override=False):
    """ A non-decorator version of memoize.
    takes a filename and a data_generator argument. """
    fullname = filename+".pkl"
    print("memoizing as", filename)
    if override or not os.path.isfile(PKL_FOLDER+fullname):
        res = data_gen()
        print("regen %s"%fullname)
        with open(PKL_FOLDER+fullname, 'wb') as fil:
            pkl.dump(res, fil)
    else:
        print("load %s"%fullname)
        with open(PKL_FOLDER+fullname, 'rb') as fil:
            res = pkl.load(fil)
    return res


def fetch(filename):
    """ grabs a previously memoized result---allowing use of file format as an output."""
    with open(PKL_FOLDER+filename+".pkl", 'rb') as fil:
        return pkl.load(fil)
